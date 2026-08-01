"""Job queue and lifecycle mixin."""

import json
import logging
import secrets
import sqlite3
import time
from typing import Any

from services_python.priority_lanes import normalize_priority_tier


class JobsMixin:
    """Mixin for DBManager."""

    def insert_job(self, job: dict[str, Any]) -> None:
        """Insert a new training job"""

        now = int(job.get("created_at", time.time()))
        total_steps = int(job.get("steps", job.get("total_steps", 100)))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, model_name, job_type, base_model, dataset_ref, description,
                    status, priority, priority_tier, total_votes, vote_weight,
                    submitter_id, org, created_ts, updated_ts, started_ts, completed_ts,
                    steps, batch_size, queue_position, estimated_start_hours,
                    active_nodes, progress_pct, current_step, total_steps, attempts,
                    latest_task_id, latest_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job["job_id"],
                    job.get("model_name", "distribai-small"),
                    job.get("job_type", "fine_tune"),
                    job.get("base_model", ""),
                    job.get("dataset_ref", ""),
                    job.get("description", ""),
                    job.get("status", "queued"),
                    int(job.get("priority", 0)),
                    normalize_priority_tier(job.get("priority_tier", "P1")),
                    int(job.get("total_votes", 0)),
                    float(job.get("vote_weight", 1.0)),
                    job.get("submitter_id", "distribai"),
                    job.get("org", "DistribAI"),
                    now,
                    now,
                    job.get("started_ts"),
                    job.get("completed_ts"),
                    total_steps,
                    int(job.get("batch_size", 32)),
                    job.get("queue_position"),
                    job.get("estimated_start_hours"),
                    int(job.get("active_nodes", 0)),
                    float(job.get("progress_pct", 0)),
                    int(job.get("current_step", 0)),
                    total_steps,
                    int(job.get("attempts", 0)),
                    job.get("latest_task_id"),
                    job.get("latest_reason"),
                ),
            )

    def create_job(
        self,
        job_type: str = "fine_tune",
        base_model: str = "",
        dataset_ref: str = "",
        hyperparams: dict[str, Any] | None = None,
        total_steps: int = 100,
        **extra: Any,
    ) -> str:
        """Create a job and its first task from REST admin/v1 handlers."""
        job_id = extra.get("job_id") or f"job_{secrets.token_urlsafe(12)}"
        job = {
            **extra,
            "job_id": job_id,
            "job_type": job_type,
            "model_name": extra.get("model_name") or base_model or "distribai-small",
            "base_model": base_model,
            "dataset_ref": dataset_ref,
            # Callers may pass an explicit batch_blob_url distinct from dataset_ref
            # (e.g. a pre-staged local/S3 batch file); only fall back to dataset_ref
            # when they didn't.
            "batch_blob_url": extra.get("batch_blob_url") or dataset_ref,
            "hparams": hyperparams or {},
            "steps": int(total_steps),
            "total_steps": int(total_steps),
        }
        self.insert_job_with_tasks(job, steps_per_task=extra.get("steps_per_task"))
        return job_id

    def create_tasks(self, job: dict[str, Any], steps_per_task: int | None = None) -> list[str]:
        total_steps = max(1, int(job.get("steps", 100)))
        chunk_size = max(
            1, min(int(steps_per_task or job.get("steps_per_task", total_steps)), total_steps)
        )
        created_ts = int(time.time())
        task_ids: list[str] = []
        with self._connect() as conn:
            offset = 0
            while offset < total_steps:
                task_steps = min(chunk_size, total_steps - offset)
                task_id = f"task_{secrets.token_urlsafe(12)}"
                task_ids.append(task_id)
                conn.execute(
                    """
                    INSERT INTO tasks (
                        task_id, job_id, assignee_node_id, status, weight_blob_url,
                        batch_blob_url, hparams_json, deadline_ts, weight_version, steps,
                        step_offset, attempt_count, max_attempts, benchmark_score,
                        reliability_score, created_ts, updated_ts
                    )
                    VALUES (?, ?, NULL, 'queued', ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        job["job_id"],
                        job.get("weight_blob_url", ""),
                        job.get("batch_blob_url", ""),
                        json.dumps(job.get("hparams", {})),
                        created_ts + int(job.get("deadline_seconds", 600)),
                        job.get("weight_version", "v0"),
                        task_steps,
                        offset,
                        int(job.get("max_attempts", 3)),
                        job.get("benchmark_score"),
                        job.get("reliability_score"),
                        created_ts,
                        created_ts,
                    ),
                )
                offset += task_steps
            conn.execute(
                """
                UPDATE jobs
                SET latest_task_id = ?, updated_ts = ?, total_steps = ?, steps = ?
                WHERE job_id = ?
                """,
                (
                    task_ids[0] if task_ids else None,
                    created_ts,
                    total_steps,
                    total_steps,
                    job["job_id"],
                ),
            )
        self.refresh_queue_positions()
        return task_ids

    def insert_job_with_tasks(
        self, job: dict[str, Any], steps_per_task: int | None = None
    ) -> list[str]:
        self.insert_job(job)
        return self.create_tasks(job, steps_per_task=steps_per_task)

    def refresh_queue_positions(self) -> None:
        queue = self.get_public_queue()
        with self._connect() as conn:
            for index, job in enumerate(queue, start=1):
                conn.execute(
                    """
                    UPDATE jobs
                    SET queue_position = ?, estimated_start_hours = ?, updated_ts = ?
                    WHERE job_id = ?
                    """,
                    (index, job["estimated_start_hours"], int(time.time()), job["job_id"]),
                )

    def get_queued_tasks(self) -> list[sqlite3.Row]:
        with self._conn_lock:
            conn = self._ensure_conn()
            cur = conn.execute(
                """
            SELECT t.*, j.priority, j.priority_tier, j.total_votes, j.vote_weight,
                   j.model_name, j.submitter_id, j.org
            FROM tasks t
            JOIN jobs j ON t.job_id = j.job_id
            WHERE t.status = 'queued'
            ORDER BY
                CASE j.priority_tier
                    WHEN 'P0' THEN 0
                    WHEN 'P1' THEN 1
                    WHEN 'P2' THEN 2
                    WHEN 'P3' THEN 3
                    ELSE 4
                END ASC,
                j.priority DESC,
                (j.total_votes * j.vote_weight) DESC,
                j.created_ts ASC,
                t.step_offset ASC
            """
            )
            return cur.fetchall()

    def get_next_available_task(self) -> sqlite3.Row | None:
        tasks = self.get_queued_tasks()
        return tasks[0] if tasks else None

    def get_queue_depth(self) -> int:
        with self._conn_lock:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM tasks WHERE status = 'queued'"
            ).fetchone()
        return int(row["count"]) if row else 0

    def get_public_queue(self) -> list[dict[str, Any]]:
        with self._conn_lock:
            conn = self._ensure_conn()
            cur = conn.execute(
                """
            SELECT job_id, description, submitter_id, org, total_votes, priority_tier,
                   status, created_ts, progress_pct, current_step, total_steps,
                   active_nodes, estimated_start_hours, priority
            FROM jobs
            WHERE status IN ('queued', 'assigned', 'running', 'success', 'failed')
            ORDER BY
                CASE priority_tier
                    WHEN 'P0' THEN 0
                    WHEN 'P1' THEN 1
                    WHEN 'P2' THEN 2
                    WHEN 'P3' THEN 3
                    ELSE 4
                END ASC,
                priority DESC,
                total_votes DESC,
                created_ts ASC
            """
            )
            rows = cur.fetchall()
        queue = []
        for index, row in enumerate(rows, start=1):
            estimated_start_hours = row["estimated_start_hours"]
            if estimated_start_hours is None:
                estimated_start_hours = round((index - 1) * 0.4, 1)
            queue.append(
                {
                    "queue_position": index,
                    "job_id": row["job_id"],
                    "description": row["description"] or "",
                    "submitter_id": row["submitter_id"],
                    "org": row["org"],
                    "total_votes": row["total_votes"] or 0,
                    "priority": row["priority_tier"],
                    "status": row["status"],
                    "estimated_start_hours": estimated_start_hours,
                    "progress_pct": row["progress_pct"] or 0.0,
                    "current_step": row["current_step"] or 0,
                    "total_steps": row["total_steps"] or 0,
                    "active_nodes": row["active_nodes"] or 0,
                }
            )
        return queue

    def update_job_aggregate(self, job_id: str, aggregate_payload: Any) -> bool:
        payload = (
            aggregate_payload
            if isinstance(aggregate_payload, str)
            else json.dumps(aggregate_payload)
        )
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE jobs
                SET aggregate_json = ?, latest_reason = ?, updated_ts = ?
                WHERE job_id = ?
                """,
                (payload, "aggregate updated", int(time.time()), job_id),
            )
            return result.rowcount > 0

    def _refresh_job_state(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        now: int,
        latest_reason: str = "",
        latest_task_id: str | None = None,
    ) -> None:
        stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) AS queued_count,
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS running_count,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
                SUM(CASE WHEN status IN ('failed', 'error', 'timeout', 'invalid_gradient', 'rejected', 'cancelled') THEN 1 ELSE 0 END) AS failed_count,
                COALESCE(SUM(CASE WHEN status = 'success' THEN steps ELSE 0 END), 0) AS completed_steps
            FROM tasks
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        total_steps = self._job_total_steps(conn, job_id)
        completed_steps = int(stats["completed_steps"] or 0)
        progress_pct = (
            100.0 if total_steps == 0 else round((completed_steps / total_steps) * 100, 2)
        )
        if stats["success_count"] == stats["total"] and stats["total"] > 0:
            status = "success"
            completed_ts = now
        elif stats["running_count"]:
            status = "running"
            completed_ts = None
        elif stats["queued_count"]:
            status = "queued"
            completed_ts = None
        elif stats["failed_count"]:
            status = "failed"
            completed_ts = now
        else:
            status = "queued"
            completed_ts = None
        conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                active_nodes = ?,
                progress_pct = ?,
                current_step = ?,
                total_steps = ?,
                completed_ts = ?,
                latest_reason = CASE WHEN ? <> '' THEN ? ELSE latest_reason END,
                latest_task_id = COALESCE(?, latest_task_id),
                updated_ts = ?
            WHERE job_id = ?
            """,
            (
                status,
                int(stats["running_count"] or 0),
                progress_pct,
                completed_steps,
                total_steps,
                completed_ts,
                latest_reason,
                latest_reason,
                latest_task_id,
                now,
                job_id,
            ),
        )

    def _job_total_steps(self, conn: sqlite3.Connection, job_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(total_steps, steps, 0) AS total_steps FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return int(row["total_steps"] or 0) if row else 0

    def operator_retry_job(self, job_id: str) -> dict[str, Any] | None:
        """Re-queue terminal tasks for an operator-initiated job retry."""
        now = int(time.time())
        terminal_task_statuses = (
            "failed",
            "error",
            "timeout",
            "cancelled",
            "success",
            "invalid_gradient",
            "rejected",
        )
        with self._connect() as conn:
            job = conn.execute(
                "SELECT job_id, latest_task_id FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not job:
                return None

            task_ids: list[str] = []
            latest = job["latest_task_id"]
            if isinstance(latest, str) and latest:
                task_ids = [latest]
            else:
                rows = conn.execute(
                    f"""
                    SELECT task_id FROM tasks
                    WHERE job_id = ? AND status IN ({",".join("?" * len(terminal_task_statuses))})
                    """,
                    (job_id, *terminal_task_statuses),
                ).fetchall()
                task_ids = [str(r["task_id"]) for r in rows]

            requeued: list[str] = []
            for task_id in task_ids:
                row = conn.execute(
                    "SELECT task_id, status FROM tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if not row or row["status"] not in terminal_task_statuses:
                    continue
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'queued',
                        assignee_node_id = NULL,
                        updated_ts = ?,
                        completed_ts = NULL,
                        last_error = 'operator retry requested'
                    WHERE task_id = ?
                    """,
                    (now, task_id),
                )
                requeued.append(task_id)

            if not requeued:
                return {"job_id": job_id, "requeued": []}

            self._refresh_job_state(
                conn,
                job_id,
                now,
                latest_reason="operator retry requested",
                latest_task_id=requeued[0],
            )
        self.refresh_queue_positions()
        return {"job_id": job_id, "requeued": requeued}

    def get_all_jobs(self) -> list[dict[str, Any]]:
        with self._conn_lock:
            conn = self._ensure_conn()
            cur = conn.execute(
                """
            SELECT job_id, model_name, job_type, base_model, dataset_ref, description,
                   status, priority, priority_tier, total_votes, vote_weight, submitter_id, org,
                   created_ts, updated_ts, started_ts, completed_ts, steps, batch_size,
                   queue_position, estimated_start_hours, active_nodes, progress_pct,
                   current_step, total_steps, attempts, latest_task_id, latest_reason
            FROM jobs
            ORDER BY created_ts DESC
            """
            )
            rows = cur.fetchall()
        return [self._job_row_to_dict(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._conn_lock:
            conn = self._ensure_conn()
            row = conn.execute(
                """
            SELECT job_id, model_name, job_type, base_model, dataset_ref, description,
                   status, priority, priority_tier, total_votes, vote_weight, submitter_id, org,
                   created_ts, updated_ts, started_ts, completed_ts, steps, batch_size,
                   queue_position, estimated_start_hours, active_nodes, progress_pct,
                   current_step, total_steps, attempts, latest_task_id, latest_reason
            FROM jobs
            WHERE job_id = ?
            """,
                (job_id,),
            ).fetchone()
            if not row:
                return None
            task_rows = conn.execute(
                """
            SELECT task_id, status, assignee_node_id, steps, step_offset, attempt_count,
                   max_attempts, deadline_ts, weight_version, gradient_blob_url, output_json, last_error
            FROM tasks
            WHERE job_id = ?
            ORDER BY step_offset ASC
            """,
                (job_id,),
            ).fetchall()
        job = self._job_row_to_dict(row)
        job["tasks"] = [
            {
                "task_id": task["task_id"],
                "status": task["status"],
                "assignee_node_id": task["assignee_node_id"],
                "steps": task["steps"],
                "step_offset": task["step_offset"],
                "attempt_count": task["attempt_count"],
                "max_attempts": task["max_attempts"],
                "deadline_ts": task["deadline_ts"],
                "weight_version": task["weight_version"],
                "gradient_blob_url": task["gradient_blob_url"],
                "output": self._safe_json(task["output_json"]),
                "last_error": task["last_error"],
            }
            for task in task_rows
        ]
        return job

    def _job_row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "job_id": row["job_id"],
            "model_name": row["model_name"],
            "job_type": row["job_type"],
            "base_model": row["base_model"],
            "dataset_ref": row["dataset_ref"],
            "description": row["description"],
            "status": row["status"],
            "priority": row["priority"],
            "priority_tier": row["priority_tier"],
            "total_votes": row["total_votes"] or 0,
            "vote_weight": float(row["vote_weight"] or 1.0),
            "submitter_id": row["submitter_id"],
            "org": row["org"],
            "created_at": row["created_ts"],
            "updated_at": row["updated_ts"],
            "started_at": row["started_ts"],
            "completed_at": row["completed_ts"],
            "steps": row["steps"],
            "batch_size": row["batch_size"],
            "queue_position": row["queue_position"],
            "estimated_start_hours": row["estimated_start_hours"],
            "active_nodes": row["active_nodes"] or 0,
            "progress_pct": row["progress_pct"] or 0.0,
            "current_step": row["current_step"] or 0,
            "total_steps": row["total_steps"] or 0,
            "attempts": row["attempts"] or 0,
            "latest_task_id": row["latest_task_id"],
            "latest_reason": row["latest_reason"],
        }

    def get_job_hparams(self, job_id: str) -> dict[str, Any]:
        """Return hparams dict from the job's preferred task ``hparams_json``."""
        with self._conn_lock:
            conn = self._ensure_conn()
            row = conn.execute(
                """
                SELECT t.hparams_json
                FROM tasks t
                LEFT JOIN jobs j ON j.job_id = t.job_id
                WHERE t.job_id = ?
                ORDER BY
                    CASE WHEN t.task_id = j.latest_task_id THEN 0 ELSE 1 END,
                    t.step_offset ASC
                LIMIT 1
                """,
                (job_id,),
            ).fetchone()
        if not row or not row["hparams_json"]:
            return {}
        try:
            parsed = json.loads(row["hparams_json"])
        except (TypeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def update_job_status(self, job_id: str, status: str, reason: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, latest_reason = ?, updated_ts = ?,
                    completed_ts = CASE WHEN ? IN ('success', 'failed', 'cancelled') THEN ? ELSE completed_ts END
                WHERE job_id = ?
                """,
                (status, reason, int(time.time()), status, int(time.time()), job_id),
            )
        if status in {"success", "failed", "cancelled", "timeout", "error"}:
            try:
                from services_python.webhook_delivery import notify_job_terminal

                notify_job_terminal(self, job_id, status, reason)
            except Exception:
                logging.getLogger(__name__).debug(
                    "callback scheduling skipped for %s", job_id, exc_info=True
                )

    def cancel_job(self, job_id: str) -> bool:
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute("SELECT job_id FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return False
            running_tasks = conn.execute(
                "SELECT assignee_node_id FROM tasks WHERE job_id = ? AND status = 'running'",
                (job_id,),
            ).fetchall()
            conn.execute(
                """
                UPDATE tasks
                SET status = 'cancelled', updated_ts = ?, completed_ts = ?, last_error = 'job cancelled'
                WHERE job_id = ? AND status IN ('queued', 'running')
                """,
                (now, now, job_id),
            )
            for running_task in running_tasks:
                if running_task["assignee_node_id"]:
                    conn.execute(
                        """
                        UPDATE active_nodes
                        SET status = 'idle', current_task_id = NULL, updated_ts = ?
                        WHERE node_id = ?
                        """,
                        (now, running_task["assignee_node_id"]),
                    )
            conn.execute(
                """
                UPDATE jobs
                SET status = 'cancelled', completed_ts = ?, updated_ts = ?, latest_reason = 'job cancelled'
                WHERE job_id = ?
                """,
                (now, now, job_id),
            )
        self.refresh_queue_positions()
        try:
            from services_python.webhook_delivery import notify_job_terminal

            notify_job_terminal(self, job_id, "cancelled", "job cancelled")
        except Exception:
            logging.getLogger(__name__).debug(
                "callback scheduling skipped for cancelled %s", job_id, exc_info=True
            )
        return True

    def clear_completed_jobs(self) -> int:
        """Delete terminal jobs and their tasks. Returns number of jobs removed."""
        terminal = ("success", "failed", "cancelled", "timeout", "error")
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT job_id FROM jobs WHERE status IN ({','.join('?' * len(terminal))})",
                terminal,
            ).fetchall()
            job_ids = [row["job_id"] for row in rows]
            if not job_ids:
                return 0
            placeholders = ",".join("?" * len(job_ids))
            conn.execute(f"DELETE FROM tasks WHERE job_id IN ({placeholders})", job_ids)
            conn.execute(f"DELETE FROM jobs WHERE job_id IN ({placeholders})", job_ids)
        return len(job_ids)
