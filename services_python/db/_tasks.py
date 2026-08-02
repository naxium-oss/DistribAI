"""Task assignment and completion mixin."""

import logging
import time
from typing import Any

from services_python.db._base import RETRYABLE_TASK_STATUSES


class TasksMixin:
    """Mixin for DBManager."""

    def assign_task(self, task_id: str, node_id: str) -> None:
        now = int(time.time())
        with self._connect() as conn:
            task = conn.execute(
                "SELECT job_id FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if not task:
                return
            conn.execute(
                """
                UPDATE tasks
                SET assignee_node_id = ?, status = 'running', attempt_count = attempt_count + 1,
                    started_ts = COALESCE(started_ts, ?), updated_ts = ?
                WHERE task_id = ?
                """,
                (node_id, now, now, task_id),
            )
            conn.execute(
                """
                UPDATE active_nodes
                SET status = 'working', current_task_id = ?, updated_ts = ?
                WHERE node_id = ?
                """,
                (task_id, now, node_id),
            )
            running_count = conn.execute(
                "SELECT COUNT(*) AS count FROM tasks WHERE job_id = ? AND status = 'running'",
                (task["job_id"],),
            ).fetchone()["count"]
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    active_nodes = ?,
                    latest_task_id = ?,
                    attempts = attempts + 1,
                    started_ts = COALESCE(started_ts, ?),
                    updated_ts = ?
                WHERE job_id = ?
                """,
                (running_count, task_id, now, now, task["job_id"]),
            )

    def record_task_progress(self, task_id: str, step: int, loss: float) -> None:
        with self._connect() as conn:
            task = conn.execute(
                "SELECT job_id, step_offset FROM tasks WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            if not task:
                return
            completed_steps = min(
                (task["step_offset"] or 0) + step, self._job_total_steps(conn, task["job_id"])
            )
            total_steps = self._job_total_steps(conn, task["job_id"])
            progress_pct = (
                100.0 if total_steps == 0 else round((completed_steps / total_steps) * 100, 2)
            )
            conn.execute(
                """
                UPDATE jobs
                SET status = 'running',
                    current_step = MAX(COALESCE(current_step, 0), ?),
                    progress_pct = MAX(COALESCE(progress_pct, 0), ?),
                    updated_ts = ?,
                    latest_reason = ?
                WHERE job_id = ?
                """,
                (
                    completed_steps,
                    progress_pct,
                    int(time.time()),
                    f"last_loss={loss:.6f}",
                    task["job_id"],
                ),
            )

    def complete_task(
        self,
        task_id: str,
        status: str,
        gradient_blob_url: str = "",
        output_json: str = "",
        reason: str = "",
    ) -> dict[str, Any] | None:
        now = int(time.time())
        with self._connect() as conn:
            task = conn.execute(
                """
                SELECT task_id, job_id, assignee_node_id, attempt_count, max_attempts
                FROM tasks
                WHERE task_id = ?
                """,
                (task_id,),
            ).fetchone()
            if not task:
                return None
            max_attempts = (
                3 if task["max_attempts"] is None else int(task["max_attempts"])
            )
            retryable = status in RETRYABLE_TASK_STATUSES and (task["attempt_count"] or 0) < max_attempts
            if retryable:
                conn.execute(
                    """
                    UPDATE tasks
                    SET assignee_node_id = NULL,
                        status = 'queued',
                        updated_ts = ?,
                        completed_ts = NULL,
                        last_error = ?,
                        gradient_blob_url = ?,
                        output_json = ?
                    WHERE task_id = ?
                    """,
                    (now, reason, gradient_blob_url, output_json, task_id),
                )
                job_status = "queued"
            else:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, updated_ts = ?, completed_ts = ?,
                        gradient_blob_url = ?, output_json = ?, last_error = ?
                    WHERE task_id = ?
                    """,
                    (status, now, now, gradient_blob_url, output_json, reason, task_id),
                )
                job_status = status
            if task["assignee_node_id"]:
                conn.execute(
                    """
                    UPDATE active_nodes
                    SET status = 'idle',
                        current_task_id = NULL,
                        jobs_completed = jobs_completed + ?,
                        jobs_failed = jobs_failed + ?,
                        updated_ts = ?
                    WHERE node_id = ?
                    """,
                    (
                        1 if status == "success" and not retryable else 0,
                        1 if status != "success" else 0,
                        now,
                        task["assignee_node_id"],
                    ),
                )
            self._refresh_job_state(
                conn, task["job_id"], now, latest_reason=reason, latest_task_id=task_id
            )
            self.refresh_queue_positions()
            result = {
                "job_id": task["job_id"],
                "task_id": task_id,
                "status": job_status,
                "requeued": retryable,
            }
        # After the connection closes, fire signed callback_url webhooks when
        # the *job* (not just this task) has reached a terminal status.
        try:
            from services_python.webhook_delivery import is_terminal_status, notify_job_terminal

            job = self.get_job(result["job_id"])
            if job and is_terminal_status(job.get("status")):
                notify_job_terminal(
                    self,
                    result["job_id"],
                    str(job.get("status")),
                    str(job.get("latest_reason") or reason or ""),
                )
        except Exception:
            logging.getLogger(__name__).debug(
                "callback scheduling skipped after task %s", task_id, exc_info=True
            )
        return result

    def update_task_progress(
        self, task_id: str, step: int, loss: float, ts: int | None = None
    ) -> None:
        """Compatibility wrapper for gRPC progress updates."""
        self.record_task_progress(task_id, step, loss)

    def update_task_result(
        self,
        task_id: str,
        node_id: str | None = None,
        status: str = "success",
        output_json: str = "",
        gradient_blob_url: str = "",
        reason: str = "",
        wall_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Compatibility wrapper for gRPC task results."""
        if node_id:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE tasks
                    SET assignee_node_id = COALESCE(assignee_node_id, ?)
                    WHERE task_id = ?
                    """,
                    (node_id, task_id),
                )
        return self.complete_task(
            task_id=task_id,
            status=status,
            gradient_blob_url=gradient_blob_url,
            output_json=output_json,
            reason=reason,
        )

    def get_job_results(self, job_id: str) -> list[dict[str, Any]]:
        with self._conn_lock:
            conn = self._ensure_conn()
            cur = conn.execute(
                """
            SELECT task_id, assignee_node_id, status, gradient_blob_url, output_json
            FROM tasks
            WHERE job_id = ? AND status IN ('success', 'failed', 'error', 'rejected')
            ORDER BY step_offset ASC
            """,
                (job_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "task_id": row["task_id"],
                "node_id": row["assignee_node_id"],
                "status": row["status"],
                "gradient_blob_url": row["gradient_blob_url"] or "",
                "output_json": row["output_json"] or "",
            }
            for row in rows
        ]

    def _recycle_task_rows(
        self,
        conn,
        rows,
        now_ts: int,
        reason: str,
        degrade_node: bool,
    ) -> list[str]:
        """Requeue running tasks (or fail them once the attempt budget is spent).

        Shared by the stale-heartbeat sweep and the immediate-disconnect path.
        Assignment already incremented ``attempt_count``, so a task whose
        count reached ``max_attempts`` has burned its whole budget and is
        marked failed instead of looping through the queue forever.
        """
        recycled: list[str] = []
        for row in rows:
            attempts = int(row["attempt_count"] or 0)
            max_attempts = 3 if row["max_attempts"] is None else int(row["max_attempts"])
            exhausted = attempts >= max_attempts
            if exhausted:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'failed',
                        assignee_node_id = NULL,
                        updated_ts = ?,
                        completed_ts = ?,
                        last_error = ?
                    WHERE task_id = ?
                    """,
                    (now_ts, now_ts, f"{reason} (attempt budget exhausted)", row["task_id"]),
                )
            else:
                conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'queued',
                        assignee_node_id = NULL,
                        updated_ts = ?,
                        last_error = ?
                    WHERE task_id = ?
                    """,
                    (now_ts, reason, row["task_id"]),
                )
            conn.execute(
                """
                UPDATE active_nodes
                SET status = ?, current_task_id = NULL, updated_ts = ?
                WHERE node_id = ?
                """,
                ("degraded" if degrade_node else "idle", now_ts, row["node_id"]),
            )
            self._refresh_job_state(
                conn,
                row["job_id"],
                now_ts,
                latest_reason="task failed permanently" if exhausted else "task requeued",
                latest_task_id=row["task_id"],
            )
            recycled.append(row["task_id"])
        return recycled

    def requeue_stale_tasks(
        self, now_ts: int | None = None, heartbeat_timeout_s: int = 30
    ) -> list[str]:
        now_ts = now_ts or int(time.time())
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.task_id, t.job_id, t.attempt_count, t.max_attempts, n.node_id
                FROM tasks t
                JOIN active_nodes n ON n.node_id = t.assignee_node_id
                WHERE t.status = 'running'
                  AND (
                        (t.deadline_ts IS NOT NULL AND t.deadline_ts < ?)
                        OR n.last_heartbeat_ts IS NULL
                        OR n.last_heartbeat_ts < ?
                  )
                """,
                (now_ts, now_ts - heartbeat_timeout_s),
            ).fetchall()
            requeued = self._recycle_task_rows(
                conn,
                rows,
                now_ts,
                reason="stale heartbeat or deadline exceeded",
                degrade_node=True,
            )
            if requeued:
                self.refresh_queue_positions()
            return requeued

    def requeue_tasks_for_node(self, node_id: str, now_ts: int | None = None) -> list[str]:
        """Immediately recycle a disconnected node's running tasks.

        Without this the work sat in ``running`` until the ~30s heartbeat
        sweep noticed, delaying every retry by the full timeout.
        """
        now_ts = now_ts or int(time.time())
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.task_id, t.job_id, t.attempt_count, t.max_attempts,
                       t.assignee_node_id AS node_id
                FROM tasks t
                WHERE t.status = 'running' AND t.assignee_node_id = ?
                """,
                (node_id,),
            ).fetchall()
            requeued = self._recycle_task_rows(
                conn,
                rows,
                now_ts,
                reason="assignee disconnected",
                degrade_node=False,
            )
            if requeued:
                self.refresh_queue_positions()
            return requeued

