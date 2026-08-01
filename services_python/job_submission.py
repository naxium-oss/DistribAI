"""Job Submission System for Related Organizations.

Handles job submissions from external orgs with custom scripts.
Supports: train, finetune, rl, inference, benchmarking, and custom scripts.
Includes distributed training support across all nodes.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import tarfile
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Database import for persistence
from .database import get_database
from .script_validation import validate_submitted_script

_SCRIPT_VALIDATION_ERROR = "submitted script failed validation"


def validate_script_file(path: str) -> tuple[list[str], list[str]]:
    """Validate an on-disk Python script before it enters a worker package."""
    script_path = Path(path)
    if not script_path.is_file():
        return ["script_path_not_found"], ["Provide an existing Python script file."]
    try:
        source = script_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"script_path_unreadable:{type(exc).__name__}"], [
            "The script file must be readable UTF-8 Python source."
        ]
    return validate_submitted_script(source)


def _validated_script_bytes(
    script_content: str | None,
    script_path: str | None,
) -> bytes:
    """Return validated script bytes or raise without creating a package."""
    if isinstance(script_content, str) and script_content.strip():
        source = script_content
    elif isinstance(script_path, str) and script_path.strip():
        workspace = os.getenv("DISTRIBAI_SCRIPT_WORKSPACE", "").strip()
        if not workspace:
            raise ValueError(
                "script_path is disabled; provide script_content or configure DISTRIBAI_SCRIPT_WORKSPACE"
            )
        path = Path(script_path)
        try:
            path.resolve().relative_to(Path(workspace).resolve())
        except (OSError, ValueError) as exc:
            raise ValueError("script_path must be inside DISTRIBAI_SCRIPT_WORKSPACE") from exc
        if not path.is_file():
            raise ValueError("script_path must point to an existing file")
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError("script_path must be readable UTF-8 Python source") from exc
    else:
        raise ValueError("job requires explicit script_content or an existing script_path")

    errors, _hints = validate_submitted_script(source)
    if errors:
        raise ValueError(f"{_SCRIPT_VALIDATION_ERROR}: {', '.join(errors)}")
    return source.encode("utf-8")


class _LazyAiohttpWeb:
    def __getattr__(self, name: str):
        from aiohttp import web as aiohttp_web

        return getattr(aiohttp_web, name)


web = _LazyAiohttpWeb()


class JobType(Enum):
    """Supported job types."""

    TRAIN = "train"
    FINETUNE = "finetune"
    RL = "rl"  # Reinforcement Learning
    INFERENCE = "inference"
    BENCHMARK = "benchmark"
    EVALUATION = "evaluation"
    CUSTOM = "custom"


class JobPriority(Enum):
    """Job priority levels."""

    CRITICAL = 0  # P0
    HIGH = 1  # P1
    NORMAL = 2  # P2
    LOW = 3  # P3
    BACKGROUND = 4  # P4


@dataclass
class JobSubmission:
    """Job submission from an organization."""

    job_id: str
    org_id: str
    job_type: JobType
    priority: JobPriority
    name: str
    description: str

    # Script/package info
    script_path: str | None = None
    script_content: str | None = None
    requirements: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    # Model/data info
    base_model: str | None = None
    dataset_ref: str | None = None
    checkpoint_url: str | None = None
    dataset_format: str = "auto"  # alpaca, sharegpt, dolly, etc.

    # Execution config
    hyperparams: dict[str, Any] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)
    total_steps: int = 1000
    min_gpu_vram_gb: float = 0.0
    required_cuda: bool = False

    # Distributed training config
    distributed_mode: bool = True  # All nodes work on same job by default
    gradient_sync_steps: int = 100  # Steps between gradient sync
    checkpoint_steps: int = 500  # Steps between checkpoints
    max_retries: int = 3  # Auto-retry on failure

    # Trainer configuration
    trainer_type: str = "distribai"  # distribai, huggingface, custom
    training_phase: str = "sft"  # pretrain, sft, rl, distill, spin

    # GRPO (Group Relative Policy Optimization) config
    # Only used when job_type == 'rl' and training_phase == 'rl'
    grpo_config: dict[str, Any] = field(default_factory=lambda: {
        "group_size": 4,
        "kl_coef": 0.1,
        "clip_eps": 0.2,
        "reward_scale": 1.0,
        "prompts_per_step": 2,
        "max_gen_tokens": 512,
        "gen_temperature": 0.9,
        "gen_top_k": 40,
    })

    # Status
    status: str = "pending"  # pending, queued, running, completed, failed, cancelled
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    submitted_by: str = ""

    execution_paradigm: str = "sync_cohort_ddp"
    cohort_id: str = ""
    federated_round_config_json: str = ""

    # Results
    result_checkpoint: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


@dataclass
class TaskAssignment:
    """Task assigned to a specific node."""

    task_id: str
    job_id: str
    job_type: JobType
    node_id: str

    # Work package
    script_package: bytes  # Tar.gz of script + deps
    work_dir: str
    start_step: int
    end_step: int
    hyperparams: dict[str, Any]
    env_vars: dict[str, str]

    # Distributed training fields
    rank: int = 0  # Node rank in distributed group (0 to world_size-1)
    world_size: int = 1  # Total number of nodes
    master_addr: str = "localhost"  # Orchestrator address for init
    master_port: int = 29500  # Port for distributed communication
    gradient_sync_steps: int = 100  # Steps between gradient synchronization

    execution_paradigm: str = "sync_cohort_ddp"
    cohort_id: str = ""
    federated_round_config_json: str = ""

    # Status
    status: str = "assigned"  # assigned, accepted, running, completed, failed
    assigned_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    result: dict | None = None
    logs: list[str] = field(default_factory=list)


class JobQueue:
    """Priority queue for job management with SQLite persistence."""

    def __init__(self):
        self._pending: list[JobSubmission] = []
        self._running: dict[str, JobSubmission] = {}
        self._completed: dict[str, JobSubmission] = {}
        self._lock = asyncio.Lock()
        self._db = get_database()

    async def submit(self, job: JobSubmission) -> str:
        """Submit a new job to the queue and persist to database."""
        async with self._lock:
            self._pending.append(job)
            # Sort by priority (lower = higher priority)
            self._pending.sort(key=lambda j: (j.priority.value, j.created_at))

            # Persist to database
            try:
                await self._db.create_job(
                    job_id=job.job_id,
                    name=job.name,
                    job_type=job.job_type.value,
                    config={
                        "org_id": job.org_id,
                        "priority": job.priority.value,
                        "description": job.description,
                        "base_model": job.base_model,
                        "dataset_ref": job.dataset_ref,
                        "dataset_format": job.dataset_format,
                        "total_steps": job.total_steps,
                        "distributed_mode": job.distributed_mode,
                        "gradient_sync_steps": job.gradient_sync_steps,
                        "checkpoint_steps": job.checkpoint_steps,
                        "max_retries": job.max_retries,
                        "trainer_type": job.trainer_type,
                        "training_phase": job.training_phase,
                        "requirements": job.requirements,
                        "execution_paradigm": job.execution_paradigm,
                        "cohort_id": job.cohort_id,
                        "federated_round_config_json": job.federated_round_config_json,
                    },
                    total_steps=job.total_steps,
                )
            except Exception as e:
                print(f"[JobQueue] Warning: Failed to persist job to database: {e}")

        return job.job_id

    async def get_next(self, node_capabilities: dict) -> JobSubmission | None:
        """Get next suitable job for a node based on capabilities."""
        async with self._lock:
            for i, job in enumerate(self._pending):
                # Check if node meets requirements
                if job.min_gpu_vram_gb > node_capabilities.get("vram_gb", 0):
                    continue
                if job.required_cuda and not node_capabilities.get("has_cuda", False):
                    continue

                # Remove from pending
                self._pending.pop(i)
                job.status = "running"
                self._running[job.job_id] = job
                return job
            return None

    async def complete(self, job_id: str, result: dict):
        """Mark job as completed."""
        async with self._lock:
            if job_id in self._running:
                job = self._running.pop(job_id)
                job.status = "completed"
                job.result_checkpoint = result.get("checkpoint")
                job.metrics = result.get("metrics", {})
                self._completed[job_id] = job

                # Update database
                try:
                    await self._db.update_job_status(
                        job_id=job_id,
                        status="completed",
                    )
                    if result.get("checkpoint"):
                        await self._db.set_job_checkpoint(job_id, result.get("checkpoint"))
                except Exception as e:
                    print(f"[JobQueue] Warning: Failed to update job status in database: {e}")

    async def fail(self, job_id: str, error: str):
        """Mark a pending or running job as failed and persist the error."""
        async with self._lock:
            job = self._running.pop(job_id, None)
            if job is None:
                for index, pending_job in enumerate(self._pending):
                    if pending_job.job_id == job_id:
                        job = self._pending.pop(index)
                        break
            if job is None:
                return

            job.status = "failed"
            job.logs.append(f"ERROR: {error}")
            self._completed[job_id] = job

            # Update database
            try:
                await self._db.update_job_status(
                    job_id=job_id,
                    status="failed",
                    error_message=error,
                )
                await self._db.add_log(
                    job_id=job_id,
                    message=error,
                    level="ERROR",
                )
            except Exception as e:
                print(f"[JobQueue] Warning: Failed to update job status in database: {e}")

    async def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        async with self._lock:
            # Check pending
            for i, job in enumerate(self._pending):
                if job.job_id == job_id:
                    job.status = "cancelled"
                    self._completed[job_id] = self._pending.pop(i)
                    return True

            # Check running
            if job_id in self._running:
                job = self._running.pop(job_id)
                job.status = "cancelled"
                self._completed[job_id] = job

                # Update database
                try:
                    await self._db.update_job_status(
                        job_id=job_id,
                        status="cancelled",
                    )
                except Exception as e:
                    print(f"[JobQueue] Warning: Failed to update job status in database: {e}")

                return True

            return False

    def get_status(self) -> dict:
        """Get queue status."""
        return {
            "pending": len(self._pending),
            "running": len(self._running),
            "completed": len(self._completed),
            "pending_jobs": [
                {"id": j.job_id, "name": j.name, "priority": j.priority.name}
                for j in self._pending[:10]  # Top 10
            ],
        }

    async def log_job_output(
        self,
        job_id: str,
        node_id: str | None,
        message: str,
        level: str = "INFO",
        stream: str = "stdout",
    ):
        """Log job output to database."""
        try:
            await self._db.add_log(
                job_id=job_id,
                node_id=node_id,
                message=message,
                level=level,
                stream=stream,
            )
        except Exception as e:
            print(f"[JobQueue] Warning: Failed to log job output: {e}")

    async def get_job_logs(self, job_id: str, limit: int = 1000) -> list[dict]:
        """Get logs for a job from database."""
        try:
            return await self._db.get_logs(job_id=job_id, limit=limit)
        except Exception as e:
            print(f"[JobQueue] Warning: Failed to get job logs: {e}")
            return []


class JobDistributor:
    """Distributes job tasks to available nodes.

    Supports two modes:
    1. Queue mode: Jobs are assigned to individual idle nodes (original behavior)
    2. Distributed mode: All nodes work on same job simultaneously (data parallelism)
    """

    def __init__(self, queue: JobQueue, node_service: Any):
        self.queue = queue
        self.node_service = node_service
        self.active_assignments: dict[str, TaskAssignment] = {}
        self._distributed_trainer = None

    def _get_distributed_trainer(self):
        """Lazy load distributed trainer."""
        if self._distributed_trainer is None:
            from .distributed_trainer import get_distributed_trainer

            self._distributed_trainer = get_distributed_trainer(self.node_service)
        return self._distributed_trainer

    async def distribute_loop(self):
        """Main distribution loop - continuously assign tasks to nodes."""
        while True:
            try:
                await self._distribute_tasks()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"[Distributor] Error: {e}")
                await asyncio.sleep(10)

    async def _distribute_tasks(self):
        """Distribute available tasks to nodes."""
        # Get idle nodes with capabilities
        idle_nodes = self._get_idle_nodes()

        if not idle_nodes:
            return

        # Get next pending job
        job = await self._get_next_pending_job()
        if not job:
            return

        is_grpo = (
            job.job_type == JobType.RL
            and job.grpo_config is not None
            and job.grpo_config.get("group_size", 0) > 0
        )
        if not is_grpo:
            try:
                _validated_script_bytes(job.script_content, job.script_path)
            except ValueError as exc:
                await self.queue.fail(job.job_id, str(exc))
                return

        if is_grpo and not job.distributed_mode:
            await self.queue.fail(
                job.job_id,
                "native GRPO jobs require distributed_mode=True",
            )
            return

        # Check if job is distributed mode (default: True)
        if job.distributed_mode:
            # Distributed mode: ALL nodes work on same job
            await self._distribute_distributed_job(job, idle_nodes)
        else:
            # Queue mode: One job per node
            for node_id, capabilities in idle_nodes:
                job = await self.queue.get_next(capabilities)
                if not job:
                    break

                task = await self._create_task(job, node_id)
                if task:
                    if not await self._send_to_node(node_id, task):
                        await self.queue.fail(job.job_id, "Failed to send task to worker")
                else:
                    await self.queue.fail(job.job_id, "Failed to package or create task")

    async def _get_next_pending_job(self) -> JobSubmission | None:
        """Get next pending job from queue."""
        # Get all pending jobs (we'll filter by capability when distributing)
        async with self.queue._lock:
            if self.queue._pending:
                return self.queue._pending[0]
        return None

    async def _distribute_distributed_job(
        self,
        job: JobSubmission,
        idle_nodes: list[tuple[str, dict]],
    ):
        """Distribute job to ALL nodes for distributed training.

        Supports both standard distributed training and GRPO (RL) jobs.

        Args:
            job: The job to run
            idle_nodes: List of (node_id, capabilities) tuples
        """
        # Check if this is a GRPO job (RL type with grpo_config)
        is_grpo = (
            job.job_type == JobType.RL
            and job.grpo_config is not None
            and job.grpo_config.get("group_size", 0) > 0
        )

        if is_grpo:
            await self._distribute_grpo_job(job, idle_nodes)
            return

        print(f"[Distributor] Starting distributed job {job.job_id} across {len(idle_nodes)} nodes")

        # Initialize distributed training
        trainer = self._get_distributed_trainer()
        success = await trainer.start_distributed_job(job, idle_nodes)

        if not success:
            print(f"[Distributor] Failed to start distributed job {job.job_id}")
            await self.queue.fail(job.job_id, "Failed to initialize distributed training")
            return

        # Create and send tasks to ALL nodes. Mark the job running before
        # assignment so an all-node packaging failure can be recorded as failed.
        if job.job_id in self.queue._pending:
            self.queue._running[job.job_id] = self.queue._pending.pop(
                self.queue._pending.index(job)
            )
            job.status = "running"

        failed_nodes = []
        successful_nodes = 0
        cooldown_sec = float(os.getenv("DISTRIBAI_JOB_ASSIGN_COOLDOWN_SEC", "0"))

        for node_id, _capabilities in idle_nodes:
            try:
                # Create task with distributed config
                task = await self._create_distributed_task(job, node_id, trainer)
                if task:
                    # Send to node
                    success = await self._send_to_node(node_id, task)
                    if not success:
                        failed_nodes.append(node_id)
                    else:
                        successful_nodes += 1
                        if cooldown_sec > 0:
                            await asyncio.sleep(cooldown_sec)
                else:
                    failed_nodes.append(node_id)
            except Exception as e:
                print(f"[Distributor] Failed to assign task to {node_id}: {e}")
                failed_nodes.append(node_id)

        if successful_nodes == 0:
            cleanup = getattr(trainer, "fail_distributed_job", None)
            if cleanup is not None:
                await cleanup(
                    job.job_id,
                    "Failed to create or assign a task on every worker",
                )
            await self.queue.fail(job.job_id, "Failed to create or assign a task on every worker")
            return

        # Log results
        if failed_nodes:
            print(
                f"[Distributor] Warning: {len(failed_nodes)} nodes failed to start job {job.job_id}"
            )

        print(
            f"[Distributor] Distributed job {job.job_id} started on {len(idle_nodes) - len(failed_nodes)}/{len(idle_nodes)} nodes"
        )

    async def _distribute_grpo_job(
        self,
        job: JobSubmission,
        idle_nodes: list[tuple[str, dict]],
    ):
        """Distribute a GRPO (RL) job using the GrpoCoordinator.

        Sends a TaskAssign to each worker (with paradigm="grpo") so it can
        initialise its GrpoRunner. Then starts the first GRPO round via
        the coordinator.
        """
        node_ids = [nid for nid, _caps in idle_nodes]
        print(
            f"[Distributor] Starting GRPO job {job.job_id} across {len(node_ids)} nodes: "
            f"group_size={job.grpo_config.get('group_size', 4)}"
        )

        # Build a send_to_worker callback using connected_nodes
        def _send_to_worker(worker_id: str, msg: Any) -> None:
            queue = self.node_service.connected_nodes.get(worker_id)
            if queue is not None:
                try:
                    # Use the existing coroutine-based queue put
                    import asyncio
                    asyncio.ensure_future(queue.put(msg))
                except Exception as exc:
                    print(f"[Distributor] Error sending to {worker_id}: {exc}")

        # Initialise the GrpoCoordinator and start the job
        from services_python.grpo_coordinator import get_grpo_coordinator

        coord = get_grpo_coordinator()
        coord.start_job(
            job_id=job.job_id,
            worker_ids=node_ids,
            grpo_config=dict(job.grpo_config),
            send_to_worker=_send_to_worker,
        )

        # Move job to running state
        if job.job_id in self.queue._pending:
            self.queue._running[job.job_id] = self.queue._pending.pop(
                self.queue._pending.index(job)
            )
            job.status = "running"

        print(
            f"[Distributor] GRPO job {job.job_id} started on {len(node_ids)} nodes via GrpoCoordinator"
        )

    def _get_idle_nodes(self) -> list[tuple[str, dict]]:
        """Get list of idle nodes with their capabilities."""
        idle = []
        if hasattr(self.node_service, "connected_nodes"):
            for node_id in self.node_service.connected_nodes:
                # Get node capabilities from metadata
                caps = self._get_node_capabilities(node_id)
                if self._is_node_idle(node_id):
                    idle.append((node_id, caps))
        return idle

    def _get_node_capabilities(self, node_id: str) -> dict:
        """Get hardware capabilities of a node."""
        # Get from node metadata
        if hasattr(self.node_service, "node_metadata"):
            meta = self.node_service.node_metadata.get(node_id, {})
            return {
                "vram_gb": meta.get("vram_gb", 0),
                "has_cuda": meta.get("has_cuda", False),
                "compute_score": meta.get("compute_score", 0),
            }
        return {"vram_gb": 0, "has_cuda": False, "compute_score": 0}

    def _is_node_idle(self, node_id: str) -> bool:
        """Check if node has no active tasks."""
        for task in self.active_assignments.values():
            if task.node_id == node_id and task.status in ["assigned", "accepted", "running"]:
                return False
        return True

    async def _create_task(self, job: JobSubmission, node_id: str) -> TaskAssignment | None:
        """Create a task assignment for a job (queue mode)."""
        try:
            # Package script
            script_package = await self._package_script(job)

            # Determine step range
            start_step = 0
            end_step = job.total_steps

            task = TaskAssignment(
                task_id=f"task-{uuid.uuid4().hex[:8]}",
                job_id=job.job_id,
                job_type=job.job_type,
                node_id=node_id,
                script_package=script_package,
                work_dir=f"/tmp/distribai/{job.job_id}",
                start_step=start_step,
                end_step=end_step,
                hyperparams=job.hyperparams,
                env_vars=job.env_vars,
                rank=0,
                world_size=1,
                execution_paradigm=job.execution_paradigm,
                cohort_id=job.cohort_id or job.job_id,
                federated_round_config_json=job.federated_round_config_json,
            )

            self.active_assignments[task.task_id] = task
            return task

        except Exception as e:
            print(f"[Distributor] Failed to create task: {e}")
            return None

    async def _create_distributed_task(
        self,
        job: JobSubmission,
        node_id: str,
        trainer: Any,
    ) -> TaskAssignment | None:
        """Create a task assignment for distributed training."""
        try:
            # Package script with distributed config
            script_package = await self._package_script(job, distributed=True)

            # Create task with distributed training fields
            task = await trainer.create_task_for_node(job, node_id, script_package)

            if task:
                self.active_assignments[task.task_id] = task

            return task

        except Exception as e:
            print(f"[Distributor] Failed to create distributed task: {e}")
            return None

    async def _package_script(self, job: JobSubmission, distributed: bool = False) -> bytes:
        """Package script and dependencies into tar.gz.

        Args:
            job: The job to package
            distributed: Whether to include distributed training config
        """
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as f:
            with tarfile.open(fileobj=f, mode="w:gz") as tar:
                # Add only validated, explicitly supplied script source.
                script_bytes = _validated_script_bytes(job.script_content, job.script_path)
                info = tarfile.TarInfo(name="run.py")
                info.size = len(script_bytes)
                tar.addfile(info, io.BytesIO(script_bytes))

                # Dependencies are supplied by the submitter; the distributor does not
                # silently add a trainer-specific environment that may not match the script.
                requirements = list(job.requirements)

                if requirements:
                    req_content = "\n".join(requirements)
                    req_bytes = req_content.encode("utf-8")
                    info = tarfile.TarInfo(name="requirements.txt")
                    info.size = len(req_bytes)
                    tar.addfile(info, io.BytesIO(req_bytes))

                # Add config.json
                config = {
                    "job_id": job.job_id,
                    "job_type": job.job_type.value,
                    "trainer_type": job.trainer_type,
                    "training_phase": job.training_phase,
                    "base_model": job.base_model,
                    "dataset_ref": job.dataset_ref,
                    "dataset_format": job.dataset_format,
                    "hyperparams": job.hyperparams,
                    "env_vars": job.env_vars,
                    "total_steps": job.total_steps,
                    "distributed_mode": distributed,
                    "grpo_config": job.grpo_config,
                }

                if distributed:
                    config["distributed"] = {
                        "gradient_sync_steps": job.gradient_sync_steps,
                        "checkpoint_steps": job.checkpoint_steps,
                        "max_retries": job.max_retries,
                    }

                config_bytes = json.dumps(config, indent=2).encode("utf-8")
                info = tarfile.TarInfo(name="config.json")
                info.size = len(config_bytes)
                tar.addfile(info, io.BytesIO(config_bytes))

            # Read back from file object (temp file auto-deleted on close)
            f.flush()
            f.seek(0)
            return f.read()

    async def _send_to_node(self, node_id: str, task: TaskAssignment) -> bool:
        """Send task assignment to a node.

        Returns:
            True if message sent successfully
        """
        from worker.src.distribai_proto import distribai_pb2

        if not hasattr(self.node_service, "connected_nodes"):
            return False
        queue = self.node_service.connected_nodes.get(node_id)
        if not queue:
            return False

        assign_msg = self._task_to_pb_assign(task)
        try:
            await queue.put(distribai_pb2.ServerMessage(assign=assign_msg))
            return True
        except Exception as e:
            print(f"[Distributor] Failed to send to {node_id}: {e}")
            return False

    def _task_to_pb_assign(self, task: TaskAssignment):
        """Build protobuf TaskAssign from a distributor task."""
        from worker.src.distribai_proto import distribai_pb2

        hparams_json = json.dumps(task.hyperparams) if task.hyperparams else "{}"
        steps = max(1, int(task.end_step - task.start_step))
        deadline_seconds = int(task.hyperparams.get("deadline_seconds", 3600))
        model_name = (
            str(task.hyperparams.get("model_name") or task.hyperparams.get("base_model") or "")
            or "custom_script"
        )
        batch_url = str(
            task.hyperparams.get("dataset_ref") or task.hyperparams.get("batch_blob_url") or ""
        )

        dist_env = dict(task.env_vars)
        dist_env.setdefault("RANK", str(task.rank))
        dist_env.setdefault("WORLD_SIZE", str(task.world_size))
        dist_env.setdefault("MASTER_ADDR", task.master_addr)
        dist_env.setdefault("MASTER_PORT", str(task.master_port))

        paradigm = task.execution_paradigm or "sync_cohort_ddp"
        cohort = task.cohort_id or task.job_id

        return distribai_pb2.TaskAssign(
            task_id=task.task_id,
            job_id=task.job_id,
            model_name=model_name[:512],
            weight_blob_url=str(task.hyperparams.get("weight_blob_url", "") or ""),
            batch_blob_url=batch_url[:2048],
            hparams_json=hparams_json,
            deadline_ts=int(time.time()) + deadline_seconds,
            weight_version=str(task.hyperparams.get("weight_version", "") or ""),
            steps=steps,
            script_package=bytes(task.script_package),
            execution_paradigm=paradigm[:128],
            cohort_id=cohort[:128],
            distributed_env_json=json.dumps(dist_env),
            federated_round_config_json=task.federated_round_config_json or "",
        )

    async def handle_task_result(self, node_id: str, result: dict):
        """Handle task result from a node."""
        task_id = result.get("task_id")
        job_id = result.get("job_id")

        if task_id in self.active_assignments:
            task = self.active_assignments[task_id]
            task.status = result.get("status", "completed")
            task.result = result
            task.completed_at = datetime.now(UTC)

            # Update job
            if task.status == "completed":
                await self.queue.complete(job_id, result)
            else:
                await self.queue.fail(job_id, result.get("error", "Unknown error"))

            # Cleanup
            del self.active_assignments[task_id]


class JobSubmissionHandler:
    """HTTP handler for job submission API."""

    def __init__(
        self,
        *,
        queue: JobQueue | None = None,
        distributor: JobDistributor | None = None,
        db: Any = None,
    ):
        self.queue = queue or job_queue  # Use global job_queue if not provided
        self.distributor = distributor
        self.db = db
        # Nothing ever populated this whitelist in practice (no admin endpoint
        # calls add_allowed_org), which made /jobs/submit permanently 403 for
        # every organization. Default to an open multi-tenant model — any
        # non-empty org_id may use the API — and let operators lock it down
        # with DISTRIBAI_ALLOWED_ORGS (comma-separated) when they need to
        # restrict access to specific orgs.
        self._explicit_allowed_orgs: set[str] = set()
        env_orgs = os.getenv("DISTRIBAI_ALLOWED_ORGS", "")
        for part in env_orgs.split(","):
            part = part.strip()
            if part:
                self._explicit_allowed_orgs.add(part)
        self._restrict_orgs = bool(self._explicit_allowed_orgs)

    def add_allowed_org(self, org_id: str):
        """Add an organization to the whitelist (also switches to restricted mode)."""
        self._explicit_allowed_orgs.add(org_id)
        self._restrict_orgs = True

    def _org_is_allowed(self, org_id: str) -> bool:
        if not org_id:
            return False
        if not self._restrict_orgs:
            return True
        return org_id in self._explicit_allowed_orgs

    @property
    def allowed_orgs(self) -> set[str]:
        """Backwards-compatible view; empty + unrestricted means "any org"."""
        return set(self._explicit_allowed_orgs)

    async def submit_job(self, req: web.Request) -> web.Response:
        """Handle job submission request."""
        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Verify organization
        org_id = body.get("org_id")
        if not org_id:
            return web.json_response({"error": "Missing org_id"}, status=400)

        if not self._org_is_allowed(org_id):
            return web.json_response({"error": "Unauthorized org"}, status=403)

        # Parse job type
        job_type_str = body.get("job_type", "train").lower()
        try:
            job_type = JobType(job_type_str)
        except ValueError:
            return web.json_response(
                {"error": f"Invalid job_type. Must be one of: {[t.value for t in JobType]}"},
                status=400,
            )

        script_content = body.get("script_content")
        script_path = body.get("script_path")
        has_script_content = isinstance(script_content, str) and bool(script_content.strip())
        has_script_path = isinstance(script_path, str) and bool(script_path.strip())
        if not has_script_content and not has_script_path:
            return web.json_response(
                {
                    "ok": False,
                    "error": "script_content or script_path is required; generated scripts are not supported",
                },
                status=400,
            )

        if has_script_content:
            err_codes, hints = validate_submitted_script(script_content)
            if err_codes:
                return web.json_response(
                    {"ok": False, "validation_errors": err_codes, "suggestions": hints},
                    status=400,
                )
        else:
            workspace = os.getenv("DISTRIBAI_SCRIPT_WORKSPACE", "").strip()
            if not workspace:
                return web.json_response(
                    {
                        "ok": False,
                        "error": "script_path is disabled for API submissions; provide script_content or configure DISTRIBAI_SCRIPT_WORKSPACE",
                    },
                    status=400,
                )
            try:
                requested = Path(script_path).resolve()
                allowed_root = Path(workspace).resolve()
                requested.relative_to(allowed_root)
            except (OSError, ValueError):
                return web.json_response(
                    {"ok": False, "error": "script_path must be inside DISTRIBAI_SCRIPT_WORKSPACE"},
                    status=400,
                )
            path_errors, path_hints = validate_script_file(script_path)
            if path_errors:
                return web.json_response(
                    {
                        "ok": False,
                        "error": _SCRIPT_VALIDATION_ERROR,
                        "validation_errors": path_errors,
                        "suggestions": path_hints,
                    },
                    status=400,
                )

        # Parse priority
        priority_str = body.get("priority", "normal").upper()
        try:
            priority = JobPriority[priority_str]
        except KeyError:
            priority = JobPriority.NORMAL

        # Create job
        job = JobSubmission(
            job_id=f"job-{uuid.uuid4().hex[:8]}",
            org_id=org_id,
            job_type=job_type,
            priority=priority,
            name=body.get("name", f"Job {job_type.value}"),
            description=body.get("description", ""),
            script_path=script_path,
            script_content=script_content,
            requirements=body.get("requirements", []),
            dependencies=body.get("dependencies", []),
            base_model=body.get("base_model"),
            dataset_ref=body.get("dataset_ref"),
            checkpoint_url=body.get("checkpoint_url"),
            dataset_format=str(body.get("dataset_format") or "auto"),
            trainer_type=str(body.get("trainer_type") or "distribai"),
            hyperparams=body.get("hyperparams", {}),
            env_vars=body.get("env_vars", {}),
            total_steps=body.get("total_steps", 1000),
            min_gpu_vram_gb=body.get("min_gpu_vram_gb", 0.0),
            required_cuda=body.get("required_cuda", False),
            submitted_by=req.headers.get("X-User-ID", "unknown"),
            execution_paradigm=str(body.get("execution_paradigm") or "sync_cohort_ddp")[:128],
            cohort_id=str(body.get("cohort_id") or "")[:128],
            federated_round_config_json=str(body.get("federated_round_config_json") or ""),
            grpo_config=body.get("grpo_config", {
                "group_size": 4,
                "kl_coef": 0.1,
                "clip_eps": 0.2,
                "reward_scale": 1.0,
                "prompts_per_step": 2,
                "max_gen_tokens": 512,
                "gen_temperature": 0.9,
                "gen_top_k": 40,
            }),
        )

        # Submit to queue
        job_id = await self.queue.submit(job)

        # Store in DB
        if self.db:
            await asyncio.to_thread(
                self.db.create_job,
                job_type=job.job_type.value,
                base_model=job.base_model or "",
                dataset_ref=job.dataset_ref or "",
                hyperparams=job.hyperparams,
                total_steps=job.total_steps,
                job_id=job.job_id,
                description=job.description,
                org=job.org_id,
                submitter_id=job.submitted_by,
            )

        return web.json_response(
            {
                "ok": True,
                "job_id": job_id,
                "status": "queued",
                "queue_position": await self._get_queue_position(job_id),
            }
        )

    async def _get_queue_position(self, job_id: str) -> int:
        """Get position in queue."""
        for i, job in enumerate(self.queue._pending):
            if job.job_id == job_id:
                return i + 1
        return 0

    async def list_jobs(self, req: web.Request) -> web.Response:
        """List jobs for an organization."""
        org_id = req.query.get("org_id")
        if not org_id:
            return web.json_response({"error": "Missing org_id"}, status=400)

        if not self._org_is_allowed(org_id):
            return web.json_response({"error": "Unauthorized"}, status=403)

        # Filter jobs by org
        jobs = []
        for job in list(self.queue._pending) + list(self.queue._running.values()):
            if job.org_id == org_id:
                jobs.append(
                    {
                        "job_id": job.job_id,
                        "name": job.name,
                        "job_type": job.job_type.value,
                        "priority": job.priority.name,
                        "status": job.status,
                        "created_at": job.created_at.isoformat(),
                        "total_steps": job.total_steps,
                    }
                )

        return web.json_response({"jobs": jobs})

    async def get_job_status(self, req: web.Request) -> web.Response:
        """Get status of a specific job."""
        job_id = req.match_info.get("job_id")
        org_id = req.query.get("org_id")

        if not self._org_is_allowed(org_id):
            return web.json_response({"error": "Unauthorized"}, status=403)

        # Search in all queues
        job = None
        for j in (
            list(self.queue._pending)
            + list(self.queue._running.values())
            + list(self.queue._completed.values())
        ):
            if j.job_id == job_id and j.org_id == org_id:
                job = j
                break

        if not job:
            return web.json_response({"error": "Job not found"}, status=404)

        return web.json_response(
            {
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status,
                "job_type": job.job_type.value,
                "priority": job.priority.name,
                "progress": self._calculate_progress(job),
                "metrics": job.metrics,
                "logs": job.logs[-100:],  # Last 100 lines
            }
        )

    def _calculate_progress(self, job: JobSubmission) -> float:
        """Calculate job progress percentage."""
        total = max(1, job.total_steps)
        metrics = job.metrics or {}

        if job.status == "completed":
            return 100.0
        if job.status in ("failed", "cancelled"):
            pct = metrics.get("progress_pct", metrics.get("progress"))
            if pct is None:
                return 0.0
            try:
                return max(0.0, min(100.0, float(pct)))
            except (TypeError, ValueError):
                return 0.0

        if job.status == "running":
            for key in ("current_step", "step", "global_step", "train_step"):
                if key not in metrics:
                    continue
                try:
                    step = float(metrics[key])
                    return max(0.0, min(100.0, 100.0 * step / total))
                except (TypeError, ValueError):
                    continue
            pct = metrics.get("progress_pct", metrics.get("progress"))
            if pct is not None:
                try:
                    return max(0.0, min(100.0, float(pct)))
                except (TypeError, ValueError):
                    pass
            return 0.0

        return max(
            0.0, min(100.0, float(metrics.get("progress_pct", metrics.get("progress", 0.0))))
        )

    def _find_active_job(self, job_id: str | None) -> JobSubmission | None:
        """Locate a pending or running job by id (None when absent)."""
        if not job_id:
            return None
        for job in list(self.queue._pending) + list(self.queue._running.values()):
            if job.job_id == job_id:
                return job
        return None

    async def cancel_job(self, req: web.Request) -> web.Response:
        """Cancel a job, but only for the organization that submitted it.

        Without the ownership check any allowed org could cancel any other
        org's queued or running jobs by guessing job ids.
        """
        job_id = req.match_info.get("job_id")
        org_id = req.query.get("org_id")

        if not self._org_is_allowed(org_id):
            return web.json_response({"error": "Unauthorized"}, status=403)

        job = self._find_active_job(job_id)
        if job is None:
            return web.json_response({"error": "Job not found or already completed"}, status=404)
        if job.org_id != org_id:
            # 404 (not 403) so foreign orgs cannot probe which job ids exist.
            return web.json_response({"error": "Job not found or already completed"}, status=404)

        success = await self.queue.cancel(job_id)
        if success:
            return web.json_response({"ok": True, "status": "cancelled"})
        return web.json_response({"error": "Job not found or already completed"}, status=404)

    async def get_queue_status(self, req: web.Request) -> web.Response:
        """Queue counters for an allowed org; job titles only for that org.

        The unauthenticated variant used to dump the names of every org's
        pending jobs. Aggregate counts remain global (useful for capacity
        planning) but per-job details are scoped to the caller's org.
        """
        org_id = req.query.get("org_id")
        if not self._org_is_allowed(org_id):
            return web.json_response({"error": "Unauthorized"}, status=403)

        status = self.queue.get_status()
        status["pending_jobs"] = [
            {"id": job.job_id, "name": job.name, "priority": job.priority.name}
            for job in self.queue._pending
            if job.org_id == org_id
        ][:10]
        return web.json_response(status)


# Global instances
job_queue = JobQueue()


def create_distributor(node_service: Any) -> JobDistributor:
    """Create job distributor."""
    return JobDistributor(job_queue, node_service)
