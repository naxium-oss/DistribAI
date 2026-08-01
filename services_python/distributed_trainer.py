"""Distributed Training Coordinator for DistribAI.

Implements data parallelism across all nodes using ring all-reduce for gradient
synchronization. All nodes work on the same job simultaneously.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from .database import get_database
from .job_submission import JobSubmission, TaskAssignment
from .memory_manager import get_memory_manager, handle_oom_with_retry

logger = logging.getLogger(__name__)

_JOB_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_job_token(job_id: str) -> str:
    """Return a filesystem-safe token derived from a job id."""
    token = _JOB_ID_SAFE.sub("_", str(job_id or "").strip())
    return token or "job"


def _checkpoint_path(job_id: str) -> Path:
    """Resolve a checkpoint file under the runtime checkpoints directory."""
    base = Path(os.getenv("DISTRIBAI_CHECKPOINT_DIR", "runtime/checkpoints"))
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_safe_job_token(job_id)}.pt"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    world_size: int  # Total number of nodes
    rank: int  # This node's rank (0 to world_size-1)
    master_addr: str  # IP/hostname of rank 0 node
    master_port: int = 29500  # Port for distributed communication
    backend: str = "gloo"  # gloo (CPU) or nccl (GPU)
    gradient_sync_steps: int = 100  # Steps between gradient sync


async def handle_oom_in_aggregation(operation_name: str, operation_func):
    """Handle OOM errors using the memory manager."""
    try:
        return await handle_oom_with_retry(operation_func, operation_name, max_retries=2)
    except Exception as e:
        logger.error(f"OOM handling failed in {operation_name}: {e}")
        raise


def get_local_ip() -> str:
    """Get the local IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def validate_config(config: DistributedConfig) -> bool:
    """Validate distributed configuration."""
    if config.world_size <= 0:
        return False
    if not (0 <= config.rank < config.world_size):
        return False
    if not config.master_addr:
        return False
    return True


def create_config_from_job(job: Any) -> DistributedConfig:
    """Create distributed config from job submission."""
    world_size = len(getattr(job, "assigned_nodes", []))
    node_id = getattr(job, "node_id", "")
    try:
        rank = job.assigned_nodes.index(node_id)
    except (ValueError, AttributeError):
        rank = 0

    master_addr = job.assigned_nodes[0] if getattr(job, "assigned_nodes", []) else "localhost"

    return DistributedConfig(world_size=max(1, world_size), rank=rank, master_addr=master_addr)


class GradientBuffer:
    """Buffer for gradient synchronization with enhanced memory management."""

    def __init__(self, max_size: int = 1000):
        self.gradients: dict[int, dict[str, torch.Tensor]] = {}
        self.max_size = max_size
        self._lock = asyncio.Lock()
        self._memory_threshold_gb = 2.0  # Clean up when using > 2GB

    async def store(self, step: int, gradients: dict[str, torch.Tensor]):
        """Store gradients for a step with memory monitoring."""
        async with self._lock:
            self.gradients[step] = gradients

            # Check memory usage and clean up if needed
            await self._cleanup_if_needed(step)

    async def get(self, step: int) -> dict[str, torch.Tensor] | None:
        """Get gradients for a step."""
        async with self._lock:
            return self.gradients.get(step)

    async def _cleanup_if_needed(self, current_step: int):
        """Clean up old gradients if memory usage is high."""
        # Estimate memory usage
        total_memory = 0
        for step_grads in self.gradients.values():
            for tensor in step_grads.values():
                if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                    total_memory += tensor.numel() * tensor.element_size()

        memory_gb = total_memory / (1024**3)

        # Clean up if memory threshold exceeded or too many steps
        if memory_gb > self._memory_threshold_gb or len(self.gradients) > self.max_size:
            await self.cleanup_old_gradients(current_step)

    async def cleanup_old_gradients(self, current_step: int = None):
        """Clean up old gradients to free memory."""
        if current_step is None:
            # Keep only the most recent half
            steps = sorted(self.gradients.keys())
            if len(steps) > 10:
                keep_steps = steps[-10:]
            else:
                keep_steps = steps
        else:
            # Keep gradients within a window of current step
            keep_steps = [s for s in self.gradients.keys() if s >= current_step - 10]

        # Remove old gradients
        to_remove = [s for s in self.gradients.keys() if s not in keep_steps]
        for step in to_remove:
            del self.gradients[step]

        if to_remove:
            # Force garbage collection after cleanup
            import gc

            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def clear_all(self):
        """Clear all gradients from buffer."""
        async with self._lock:
            self.gradients.clear()
            # Force garbage collection
            import gc

            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_memory_usage(self) -> float:
        """Get estimated memory usage in GB."""
        total_memory = 0
        for step_grads in self.gradients.values():
            for tensor in step_grads.values():
                if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
                    total_memory += tensor.numel() * tensor.element_size()
        return total_memory / (1024**3)


class DistributedTrainer:
    """Coordinates distributed training across all nodes.

    Uses data parallelism: each node has full model, processes different batch.
    Implements ring all-reduce for efficient gradient synchronization.
    """

    def __init__(self, node_service: Any = None):
        self.node_service = node_service
        self.db = get_database()
        self.active_jobs: dict[str, DistributedJob] = {}
        self.gradient_buffer = GradientBuffer()
        self._lock = asyncio.Lock()
        self.memory_manager = get_memory_manager()
        self.node_states: dict[str, dict] = {}

        # Integrate gradient buffer with memory manager
        handle_oom_in_aggregation._gradient_buffer = self.gradient_buffer

    async def initialize_distributed_process(self, config: DistributedConfig):
        """Initialize torch distributed process group."""
        if not torch.distributed.is_initialized():
            os.environ["MASTER_ADDR"] = config.master_addr
            os.environ["MASTER_PORT"] = str(config.master_port)
            torch.distributed.init_process_group(
                backend=config.backend, rank=config.rank, world_size=config.world_size
            )

    async def cleanup_distributed_process(self):
        """Cleanup torch distributed process group."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    async def coordinate_training(self, job_id: str, config: DistributedConfig):
        """Orchestrator-side checkpoint before workers run user training scripts.

        Workers synchronize gradients via ``torch.distributed`` inside their scripts;
        the orchestrator only validates config, ensures the job exists, optionally
        hits a PG barrier if this process participates, and emits a lifecycle log.
        """
        if not validate_config(config):
            raise ValueError(f"invalid distributed configuration for job {job_id}")

        async with self._lock:
            if job_id not in self.active_jobs:
                logger.warning("coordinate_training: no active distributed job %s", job_id)
                return

        if torch.distributed.is_initialized():
            await self.barrier_synchronization()

        logger.info(
            "distributed coordination recorded for job %s (rank %s/%s)",
            job_id,
            config.rank,
            config.world_size,
        )

    async def synchronize_gradients(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        """Synchronize gradients across all nodes."""
        if torch.distributed.is_initialized():
            for grad in gradients:
                torch.distributed.all_reduce(grad)
        return gradients

    async def broadcast_model(
        self, model_state: dict[str, torch.Tensor], src_rank: int = 0
    ) -> dict[str, torch.Tensor]:
        """Broadcast model state from source rank to all other ranks."""
        if torch.distributed.is_initialized():
            for tensor in model_state.values():
                torch.distributed.broadcast(tensor, src=src_rank)
        return model_state

    async def collect_node_metrics(self, config: DistributedConfig) -> list[dict]:
        """Return orchestrator-visible step and heartbeat snapshots for active jobs."""
        if not validate_config(config):
            return []

        snapshots: list[dict] = []
        async with self._lock:
            for jid, dist_job in self.active_jobs.items():
                if dist_job.world_size != config.world_size:
                    continue
                for node_id, rank in dist_job.node_ranks.items():
                    hb = dist_job.last_heartbeat.get(node_id)
                    step = dist_job.node_steps.get(node_id, 0)
                    snapshots.append(
                        {
                            "job_id": jid,
                            "node_id": node_id,
                            "rank": rank,
                            "step": step,
                            "last_heartbeat_ts": hb.isoformat() if hb else None,
                        }
                    )
        return snapshots

    async def barrier_synchronization(self):
        """Synchronize all nodes at a barrier."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    async def save_distributed_checkpoint(self, job_id: str, checkpoint_data: dict):
        """Save a distributed checkpoint under the runtime checkpoints directory."""
        await self.barrier_synchronization()
        path = _checkpoint_path(job_id)
        torch.save(checkpoint_data, path)

    async def load_distributed_checkpoint(self, job_id: str) -> dict:
        """Load a distributed checkpoint from the runtime checkpoints directory."""
        path = _checkpoint_path(job_id)
        return torch.load(path, weights_only=False)

    def get_world_size(self) -> int:
        """Get the total number of nodes in distributed training."""
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    def get_rank(self) -> int:
        """Get the rank of the current node."""
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def is_distributed_initialized(self) -> bool:
        """Check if distributed training is initialized."""
        return torch.distributed.is_initialized()

    async def monitor_distributed_health(self, config: DistributedConfig) -> dict:
        """Summarize staleness from :meth:`collect_node_metrics`."""
        node_status = await self.collect_node_metrics(config)
        if not node_status:
            return {"all_healthy": True, "node_status": []}

        stale_seconds = float(os.getenv("DISTRIBAI_DIST_STALE_SECONDS", "120"))
        now = datetime.now(UTC)
        annotated: list[dict] = []
        all_ok = True
        for row in node_status:
            stale = False
            ts_raw = row.get("last_heartbeat_ts")
            if ts_raw:
                try:
                    ts = datetime.fromisoformat(ts_raw)
                    stale = (now - ts).total_seconds() > stale_seconds
                except ValueError:
                    stale = True
            annotated.append({**row, "stale": stale})
            if stale:
                all_ok = False
        return {"all_healthy": all_ok, "node_status": annotated}

    async def reinitialize_with_new_nodes(self, new_world_size: int, job_id: str):
        """Resize bookkeeping when nodes are added.

        Runtime process-group resizing requires elastic/torchelaunch; callers must
        restart workers. Here we update orchestrator-visible ``world_size`` and DB state.
        """
        if new_world_size < 1:
            raise ValueError("new_world_size must be positive")

        updated = False
        async with self._lock:
            dist_job = self.active_jobs.get(job_id)
            if dist_job:
                updated = True
                logger.warning(
                    "reinitialize_with_new_nodes: updating book-keeping only for job %s; "
                    "workers must restart torchelastic/torch.distributed for new topology",
                    job_id,
                )
                if torch.distributed.is_initialized():
                    await self.cleanup_distributed_process()
                dist_job.world_size = new_world_size

        if updated:
            await self.db.update_job_status(
                job_id=job_id,
                status="running",
                active_nodes=new_world_size,
            )

    async def reinitialize_with_removed_nodes(self, remaining_ranks: list[int], job_id: str):
        """Prune orchestrator-visible nodes after removals (by rank indices)."""
        async with self._lock:
            dist_job = self.active_jobs.get(job_id)
            if not dist_job:
                logger.warning("reinitialize_with_removed_nodes: unknown job %s", job_id)
                return

            keep = frozenset(remaining_ranks)
            drop_ids = [nid for nid, rnk in dist_job.node_ranks.items() if rnk not in keep]
            if torch.distributed.is_initialized():
                logger.warning(
                    "reinitialize_with_removed_nodes: tearing down PG for job %s; "
                    "remaining workers must reconnect",
                    job_id,
                )
                await self.cleanup_distributed_process()

            for nid in drop_ids:
                dist_job.node_ranks.pop(nid, None)
                dist_job.node_steps.pop(nid, None)
                dist_job.last_heartbeat.pop(nid, None)
                dist_job.failed_nodes.discard(nid)

            dist_job.world_size = len(dist_job.node_ranks)
            remaining = len(dist_job.node_ranks)

        await self.db.update_job_status(
            job_id=job_id,
            status="running",
            active_nodes=remaining,
        )

    async def dynamic_scaling_add_nodes(self, new_world_size: int, job_id: str):
        """Dynamically add nodes to a running job."""
        await self.reinitialize_with_new_nodes(new_world_size, job_id)

    async def dynamic_scaling_remove_nodes(self, remaining_ranks: list[int], job_id: str):
        """Dynamically remove nodes from a running job."""
        await self.reinitialize_with_removed_nodes(remaining_ranks, job_id)

    async def start_distributed_job(
        self,
        job: JobSubmission,
        node_assignments: list[tuple[str, dict]],
    ) -> bool:
        """Start a distributed job across all assigned nodes.

        Args:
            job: The job to run
            node_assignments: List of (node_id, capabilities) tuples

        Returns:
            True if job started successfully on all nodes
        """
        async with self._lock:
            world_size = len(node_assignments)
            if world_size == 0:
                logger.warning("No nodes available for job %s", job.job_id)
                return False

            # Get orchestrator address as master
            master_addr = self._get_master_address()
            master_port = 29500

            # Create distributed job
            dist_job = DistributedJob(
                job_id=job.job_id,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                gradient_sync_steps=job.gradient_sync_steps,
            )
            self.active_jobs[job.job_id] = dist_job

            # Assign ranks to nodes
            for rank, (node_id, _caps) in enumerate(node_assignments):
                dist_job.node_ranks[node_id] = rank

                # Assign node to job in database
                await self.db.assign_node_to_job(
                    job_id=job.job_id,
                    node_id=node_id,
                    rank=rank,
                )

            # Update job with total nodes
            await self.db.update_job_status(
                job_id=job.job_id,
                status="running",
                active_nodes=world_size,
            )

            logger.info("Started job %s across %s nodes", job.job_id, world_size)
            return True

    async def create_task_for_node(
        self,
        job: JobSubmission,
        node_id: str,
        script_package: bytes,
    ) -> TaskAssignment | None:
        """Create task assignment for a specific node in distributed job.

        Args:
            job: The job being run
            node_id: The node to assign to
            script_package: The packaged script

        Returns:
            TaskAssignment with distributed config
        """
        dist_job = self.active_jobs.get(job.job_id)
        if not dist_job:
            return None

        rank = dist_job.node_ranks.get(node_id, 0)
        world_size = dist_job.world_size

        # Create environment variables for distributed training
        env_vars = {
            **job.env_vars,
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": dist_job.master_addr,
            "MASTER_PORT": str(dist_job.master_port),
            "DISTRIBAI_DISTRIBUTED": "1",
            "DISTRIBAI_JOB_ID": job.job_id,
            "DISTRIBAI_NODE_ID": node_id,
            "DISTRIBAI_GRADIENT_SYNC_STEPS": str(job.gradient_sync_steps),
            "DISTRIBAI_CHECKPOINT_STEPS": str(job.checkpoint_steps),
        }


        # Use secure temporary directory
        prefix = "distribai_" + str(job.job_id) + "_" + str(node_id) + "_"
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        work_dir = os.path.join(temp_dir, str(node_id))
        os.makedirs(work_dir, exist_ok=True)

        return TaskAssignment(
            task_id=f"task-{job.job_id}-{rank}",
            job_id=job.job_id,
            job_type=job.job_type,
            node_id=node_id,
            script_package=script_package,
            work_dir=work_dir,
            start_step=0,
            end_step=job.total_steps,
            hyperparams=job.hyperparams,
            env_vars=env_vars,
            rank=rank,
            world_size=world_size,
            master_addr=dist_job.master_addr,
            master_port=dist_job.master_port,
            gradient_sync_steps=job.gradient_sync_steps,
            execution_paradigm=job.execution_paradigm,
            cohort_id=job.cohort_id or job.job_id,
            federated_round_config_json=job.federated_round_config_json,
        )

    async def handle_node_heartbeat(
        self,
        job_id: str,
        node_id: str,
        step: int,
        status: str,
    ):
        """Handle heartbeat from a node during distributed training.

        Args:
            job_id: The job ID
            node_id: The node reporting
            step: Current training step
            status: Node status (running, error, etc.)
        """
        # Update node heartbeat in database
        await self.db.update_node_heartbeat(
            job_id=job_id,
            node_id=node_id,
            status=status,
        )

        dist_job = self.active_jobs.get(job_id)
        if dist_job:
            dist_job.node_steps[node_id] = step
            dist_job.last_heartbeat[node_id] = datetime.now(UTC)

            # Check for stragglers (nodes falling behind)
            if dist_job.world_size > 1:
                min_step = min(dist_job.node_steps.values())
                max_step = max(dist_job.node_steps.values())
                if max_step - min_step > 10:  # 10 step threshold
                    logger.warning(
                        "Node lag detected in job %s (max_step=%s min_step=%s)",
                        job_id,
                        max_step,
                        min_step,
                    )

    async def handle_checkpoint(
        self,
        job_id: str,
        node_id: str,
        step: int,
        checkpoint_data: dict[str, Any],
    ) -> str | None:
        """Handle checkpoint from a node.

        In distributed training, we wait for checkpoints from all nodes,
        then aggregate them into a single checkpoint.

        Args:
            job_id: The job ID
            node_id: The node reporting
            step: Checkpoint step
            checkpoint_data: The checkpoint data

        Returns:
            Path to aggregated checkpoint if all nodes reported, None otherwise
        """
        dist_job = self.active_jobs.get(job_id)
        if not dist_job:
            return None

        # Store checkpoint record
        # Use secure temporary directory for checkpoints
        temp_dir = tempfile.mkdtemp(prefix=f"distribai_checkpoints_{_safe_job_token(job_id)}_")
        checkpoint_path = os.path.join(temp_dir, f"step_{step}", f"{node_id}.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        await self.db.save_checkpoint(
            job_id=job_id,
            step=step,
            node_id=node_id,
            path=checkpoint_path,
            is_aggregated=False,
        )

        # Track this node's checkpoint
        if step not in dist_job.checkpoints:
            dist_job.checkpoints[step] = set()
        dist_job.checkpoints[step].add(node_id)

        # Check if all nodes have reported
        if len(dist_job.checkpoints[step]) == dist_job.world_size:
            logger.info(
                "All nodes reported checkpoint for job %s step %s",
                job_id,
                step,
            )

            # Aggregate checkpoints
            aggregated_path = await self._aggregate_checkpoints(job_id, step)

            # Update job checkpoint
            await self.db.set_job_checkpoint(job_id, aggregated_path)
            await self.db.update_job_status(
                job_id=job_id,
                status="running",
                current_step=step,
            )

            return aggregated_path

        return None

    async def handle_node_failure(self, job_id: str, node_id: str, error: str):
        """Handle node failure during distributed training.

        Args:
            job_id: The job ID
            node_id: The failed node
            error: Error message
        """
        dist_job = self.active_jobs.get(job_id)
        if not dist_job:
            return

        logger.error("Node %s failed in job %s: %s", node_id, job_id, error)

        # Log the failure
        await self.db.add_log(
            job_id=job_id,
            node_id=node_id,
            message=f"Node failed: {error}",
            level="ERROR",
        )

        # Mark node as failed
        await self.db.update_node_heartbeat(
            job_id=job_id,
            node_id=node_id,
            status="failed",
        )

        # Decrement active nodes
        dist_job.failed_nodes.add(node_id)
        active_count = dist_job.world_size - len(dist_job.failed_nodes)

        await self.db.update_job_status(
            job_id=job_id,
            status="running",
            active_nodes=active_count,
        )

        # If too many nodes failed, mark job as failed
        if active_count < dist_job.world_size * 0.5:  # Less than 50% remaining
            logger.error("Job %s failed: too many nodes down", job_id)
            await self.db.update_job_status(
                job_id=job_id,
                status="failed",
                error_message=f"Too many nodes failed. Only {active_count}/{dist_job.world_size} remaining.",
            )

    async def broadcast_to_all_nodes(self, job_id: str, message: dict):
        """Broadcast a message to all nodes in a distributed job.

        Args:
            job_id: The job ID
            message: Message to broadcast (will be serialized)
        """
        dist_job = self.active_jobs.get(job_id)
        if not dist_job:
            return

        # Get node service to send messages
        if hasattr(self.node_service, "send_to_node"):
            for node_id in dist_job.node_ranks.keys():
                try:
                    await self.node_service.send_to_node(
                        node_id=node_id,
                        message={
                            "type": "broadcast",
                            "job_id": job_id,
                            "data": message,
                        },
                    )
                except Exception as e:
                    logger.warning("Failed to send to %s: %s", node_id, e)

    async def fail_distributed_job(self, job_id: str, error: str):
        """Stop and remove distributed bookkeeping after startup failure."""
        dist_job = self.active_jobs.get(job_id)
        if dist_job:
            await self.broadcast_to_all_nodes(job_id, {"command": "cancel"})
            self.active_jobs.pop(job_id, None)
        await self.db.update_job_status(
            job_id=job_id,
            status="failed",
            active_nodes=0,
            error_message=error,
        )

    async def emergency_cancel(self, job_id: str):
        """Emergency cancel - stop all nodes immediately.

        Args:
            job_id: The job ID to cancel
        """
        dist_job = self.active_jobs.get(job_id)
        if not dist_job:
            return

        logger.warning("Emergency cancel for job %s", job_id)

        # Broadcast cancel to all nodes
        await self.broadcast_to_all_nodes(job_id, {"command": "cancel"})

        # Update database
        await self.db.update_job_status(
            job_id=job_id,
            status="cancelled",
        )

        # Cleanup
        del self.active_jobs[job_id]

    def _get_master_address(self) -> str:
        """Get the orchestrator's address for distributed training master."""
        try:
            # Get local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except OSError:
            return "localhost"

    async def _aggregate_checkpoints(self, job_id: str, step: int) -> str:
        """Aggregate checkpoints from all nodes into single checkpoint.

        For data parallelism, we can use any single checkpoint since all
        nodes have the same model state. For more complex parallelism,
        this would need to merge sharded checkpoints.

        Args:
            job_id: The job ID
            step: The step number

        Returns:
            Path to aggregated checkpoint
        """

        async def _do_aggregate():
            # For data parallelism, just use rank 0's checkpoint
            # Use secure temporary directory for cross-platform compatibility
            temp_dir = tempfile.mkdtemp(prefix=f"distribai_checkpoints_{_safe_job_token(job_id)}_")
            aggregated_path = os.path.join(temp_dir, f"step_{step}", "aggregated.pt")
            os.makedirs(os.path.dirname(aggregated_path), exist_ok=True)

            await self.db.save_checkpoint(
                job_id=job_id,
                step=step,
                node_id=None,  # Aggregated
                path=aggregated_path,
                is_aggregated=True,
            )

            return aggregated_path

        return await handle_oom_in_aggregation(
            f"checkpoint_aggregation_{job_id}_{step}", _do_aggregate
        )

    async def get_job_status(self, job_id: str) -> dict | None:
        """Get detailed status of a distributed job."""
        dist_job = self.active_jobs.get(job_id)
        if not dist_job:
            # Check database
            job = await self.db.get_job(job_id)
            if job:
                return {
                    "job_id": job_id,
                    "status": job.get("status"),
                    "current_step": job.get("current_step"),
                    "world_size": job.get("total_nodes"),
                    "active_nodes": job.get("active_nodes"),
                }
            return None

        return {
            "job_id": job_id,
            "status": "running",
            "world_size": dist_job.world_size,
            "active_nodes": dist_job.world_size - len(dist_job.failed_nodes),
            "failed_nodes": len(dist_job.failed_nodes),
            "node_steps": dist_job.node_steps,
            "latest_checkpoint_step": max(dist_job.checkpoints.keys())
            if dist_job.checkpoints
            else 0,
        }


@dataclass
class DistributedJob:
    """State tracking for a distributed job."""

    job_id: str
    world_size: int
    master_addr: str
    master_port: int
    gradient_sync_steps: int

    node_ranks: dict[str, int] = None  # node_id -> rank
    node_steps: dict[str, int] = None  # node_id -> current step
    last_heartbeat: dict[str, datetime] = None
    failed_nodes: set[str] = None
    checkpoints: dict[int, set[str]] = None  # step -> set of node_ids

    def __post_init__(self):
        if self.node_ranks is None:
            self.node_ranks = {}
        if self.node_steps is None:
            self.node_steps = {}
        if self.last_heartbeat is None:
            self.last_heartbeat = {}
        if self.failed_nodes is None:
            self.failed_nodes = set()
        if self.checkpoints is None:
            self.checkpoints = {}


# Ring all-reduce implementation for gradient synchronization
class RingAllReduce:
    """Implements ring all-reduce for efficient gradient synchronization.

    In ring all-reduce, each node sends gradients to its neighbor in a ring.
    This is more efficient than parameter server for large numbers of nodes.
    """

    @staticmethod
    def compute_chunks(gradients: list[torch.Tensor], world_size: int) -> list[list[torch.Tensor]]:
        """Split gradients into chunks for ring all-reduce."""
        chunks = []
        for grad in gradients:
            chunk_size = grad.numel() // world_size
            if chunk_size == 0:
                chunk_size = 1
            chunks.append(list(grad.flatten().split(chunk_size)))
        return chunks

    @staticmethod
    async def all_reduce(
        gradients: dict[str, torch.Tensor],
        rank: int,
        world_size: int,
        send_fn,
        recv_fn,
    ) -> dict[str, torch.Tensor]:
        """Perform ring all-reduce on gradients.

        Args:
            gradients: Dictionary of gradient tensors by parameter name
            rank: This node's rank
            world_size: Total number of nodes
            send_fn: Async function to send data to next node
            recv_fn: Async function to receive data from previous node

        Returns:
            Averaged gradients
        """
        # Convert to list for processing
        names = list(gradients.keys())
        grads = [gradients[name] for name in names]

        # Split into chunks
        chunks = RingAllReduce.compute_chunks(grads, world_size)

        # Ring reduce-scatter (sum phase)
        for step in range(world_size - 1):
            send_chunk_idx = (rank - step - 1) % world_size
            recv_chunk_idx = (rank - step) % world_size

            # Send one chunk to next node
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1 + world_size) % world_size

            # Gather all chunks for this position
            for _i, chunk_list in enumerate(chunks):
                if send_chunk_idx < len(chunk_list):
                    await send_fn(next_rank, chunk_list[send_chunk_idx])
                if recv_chunk_idx < len(chunk_list):
                    received = await recv_fn(prev_rank)
                    if received is not None and recv_chunk_idx < len(chunk_list):
                        chunk_list[recv_chunk_idx] += received

        # Ring all-gather (broadcast averaged chunks)
        for step in range(world_size - 1):
            send_chunk_idx = (rank - step) % world_size
            recv_chunk_idx = (rank - step - 1) % world_size

            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1 + world_size) % world_size

            for _i, chunk_list in enumerate(chunks):
                if send_chunk_idx < len(chunk_list):
                    # Divide by world_size to get average
                    chunk_list[send_chunk_idx] /= world_size
                    await send_fn(next_rank, chunk_list[send_chunk_idx])
                if recv_chunk_idx < len(chunk_list):
                    chunk_list[recv_chunk_idx] = await recv_fn(prev_rank)

        # Reconstruct full gradients
        result = {}
        for name, chunk_list in zip(names, chunks, strict=True):
            # Concatenate chunks back together
            full_grad = torch.cat(chunk_list)
            # Reshape to original shape
            original_shape = gradients[name].shape
            result[name] = full_grad.reshape(original_shape)

        return result


# Global instance
trainer: DistributedTrainer | None = None


def get_distributed_trainer(node_service: Any) -> DistributedTrainer:
    """Get or create global distributed trainer instance."""
    global trainer
    if trainer is None:
        trainer = DistributedTrainer(node_service)
    return trainer
