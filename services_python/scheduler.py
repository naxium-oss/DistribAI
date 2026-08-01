"""Task scheduling logic for DistribAI orchestrator.

This module contains the task scheduling and assignment logic,
separated from the main service for better testability and maintainability.
"""

import asyncio
import json
import logging
import time
from typing import Any

from services_python.bundle_store import load_bundle
from services_python.constants import DEFAULT_REQUEUE_HEARTBEAT_TIMEOUT_SECONDS
from services_python.db_manager import DBManager

# Forward declaration for type hints
NodeService = None

logger = logging.getLogger(__name__)


class TaskScheduler:
    """Scheduler for distributing tasks to idle worker nodes."""

    def __init__(
        self,
        db: DBManager,
        node_service: "NodeService",
        check_interval: float = 0.5,
    ) -> None:
        self.db = db
        self.node_service = node_service
        self.check_interval = check_interval
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Task scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Task scheduler stopped")

    async def _loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._process_stale_tasks()
                await self._assign_queued_tasks()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Scheduler loop error: %s", exc)

            await asyncio.sleep(self.check_interval)

    async def _process_stale_tasks(self) -> None:
        """Requeue tasks from nodes that have timed out."""
        requeued = await asyncio.to_thread(
            self.db.requeue_stale_tasks,
            int(time.time()),
            DEFAULT_REQUEUE_HEARTBEAT_TIMEOUT_SECONDS,
        )
        if requeued:
            logger.warning("Requeued stale tasks: %s", ", ".join(requeued))

    async def _assign_queued_tasks(self) -> None:
        """Assign queued tasks to idle nodes."""
        tasks = await asyncio.to_thread(self.db.get_queued_tasks)

        for task in tasks:
            node_id = await asyncio.to_thread(self._find_idle_node)
            if not node_id:
                break

            await self._assign_task_to_node(task, node_id)

    def _find_idle_node(self) -> str | None:
        """Find an idle connected node."""
        nodes = {node["node_id"]: node for node in self.db.get_all_nodes()}

        for node_id in self.node_service.connected_nodes:
            node = nodes.get(node_id)
            if not node:
                continue
            if node.get("contributing", True) is False:
                continue
            if node_id in self.node_service.pending_assignments:
                continue
            return node_id

        return None

    async def _assign_task_to_node(self, task: dict[str, Any], node_id: str) -> None:
        """Assign a task to a specific node."""
        from worker.src.distribai_proto import distribai_pb2

        task = dict(task)
        await asyncio.to_thread(self.db.assign_task, task["task_id"], node_id)

        weight_url = task.get("weight_blob_url") or self.node_service.generate_presigned_url(
            f"weights/{task['job_id']}.pt"
        )
        hparams_json = task.get("hparams_json")
        if not hparams_json:
            hyperparams = task.get("hyperparams", {})
            if isinstance(hyperparams, str):
                hparams_json = hyperparams
            else:
                hparams_json = json.dumps(hyperparams)

        script_package = b""
        execution_paradigm = "legacy_builtin"
        packages = getattr(self.node_service, "script_packages", None)
        if isinstance(packages, dict):
            script_package = bytes(packages.get(task["task_id"], b""))
        if not script_package:
            loaded = await asyncio.to_thread(load_bundle, task["task_id"])
            if loaded:
                script_package = loaded
        if script_package:
            execution_paradigm = "script"
            try:
                parsed_hparams = json.loads(hparams_json) if isinstance(hparams_json, str) else {}
            except json.JSONDecodeError:
                parsed_hparams = {}
            if not isinstance(parsed_hparams, dict):
                parsed_hparams = {}
            task_context = {
                "distribai_job_id": task["job_id"],
                "distribai_task_id": task["task_id"],
                "distribai_task_steps": int(task.get("steps", 25)),
                "distribai_step_offset": int(task.get("step_offset") or 0),
            }
            parsed_hparams.update(task_context)
            parsed_hparams.setdefault("steps", task_context["distribai_task_steps"])
            hparams_json = json.dumps(parsed_hparams)
        else:
            try:
                parsed = json.loads(hparams_json) if isinstance(hparams_json, str) else {}
                if isinstance(parsed, dict):
                    paradigm = parsed.get("execution_paradigm")
                    if isinstance(paradigm, str) and paradigm:
                        execution_paradigm = paradigm
            except json.JSONDecodeError:
                pass

        assign_msg = distribai_pb2.TaskAssign(
            task_id=task["task_id"],
            job_id=task["job_id"],
            model_name=task.get("model_name") or task.get("base_model") or "distribai-small",
            weight_blob_url=weight_url or "",
            batch_blob_url=task.get("batch_blob_url") or task.get("dataset_ref", ""),
            hparams_json=hparams_json,
            deadline_ts=int(task.get("deadline_ts") or (int(time.time()) + 600)),
            weight_version=task.get("weight_version", ""),
            steps=task.get("steps", 25),
            script_package=script_package,
            execution_paradigm=execution_paradigm,
            cohort_id="",
            distributed_env_json=json.dumps(
                {
                    "DISTRIBAI_JOB_ID": task["job_id"],
                    "DISTRIBAI_TASK_ID": task["task_id"],
                    "DISTRIBAI_TASK_STEPS": str(task.get("steps", 25)),
                    "DISTRIBAI_STEP_OFFSET": str(task.get("step_offset") or 0),
                }
            )
            if script_package
            else "{}",
            federated_round_config_json="",
        )

        queue = self.node_service.connected_nodes.get(node_id)
        if queue:
            await queue.put(distribai_pb2.ServerMessage(assign=assign_msg))
            self.node_service.pending_assignments[node_id] = task["task_id"]
            logger.info("Assigned task %s to node %s", task["task_id"], node_id)
