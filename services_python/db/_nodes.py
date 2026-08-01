"""Node registration and heartbeat mixin."""

import re
import time
from typing import Any


class NodesMixin:
    """Mixin for DBManager."""

    def register_node(
        self,
        node_id: str,
        session_token: str,
        hw_json: str,
        ts: int,
        benchmark_json: str | None = None,
        jwt_token: str | None = None,
    ) -> None:
        """Register a new compute node."""
        benchmark_json = benchmark_json or ""
        jwt_token = jwt_token or ""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO active_nodes (
                    node_id, session_token, jwt_token, hardware_json, benchmark_json,
                    status, contributing, current_task_id, last_heartbeat_ts,
                    jobs_completed, jobs_failed, reliability_score, created_ts, updated_ts
                )
                VALUES (?, ?, ?, ?, ?, 'idle', 1, NULL, ?, 0, 0, 1.0, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    session_token=excluded.session_token,
                    jwt_token=CASE
                        WHEN excluded.jwt_token IS NOT NULL AND excluded.jwt_token <> '' THEN excluded.jwt_token
                        ELSE active_nodes.jwt_token
                    END,
                    hardware_json=excluded.hardware_json,
                    benchmark_json=excluded.benchmark_json,
                    status='idle',
                    last_heartbeat_ts=excluded.last_heartbeat_ts,
                    updated_ts=excluded.updated_ts
                """,
                (node_id, session_token, jwt_token, hw_json, benchmark_json, ts, ts, ts),
            )

    def create_node(self, node_id: str, jwt_token: str, hardware_json: str = "{}") -> str:
        """Create or update a node from REST registration paths."""
        self.register_node(
            node_id=node_id,
            session_token=jwt_token,
            hw_json=hardware_json,
            ts=int(time.time()),
            jwt_token=jwt_token,
        )
        return node_id

    def update_node_hardware(
        self, node_id: str, hardware_json: str, benchmark_json: str | None = None
    ) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE active_nodes
                SET hardware_json = ?,
                    benchmark_json = COALESCE(?, benchmark_json),
                    updated_ts = ?
                WHERE node_id = ?
                """,
                (hardware_json, benchmark_json, int(time.time()), node_id),
            )
            return result.rowcount > 0

    def get_node_jwt(self, node_id: str) -> str | None:
        """Return the last JWT issued for ``node_id``, or None if unknown."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT jwt_token FROM active_nodes WHERE node_id = ? LIMIT 1",
                (node_id,),
            ).fetchone()
            if not row:
                return None
            value = row[0] if not hasattr(row, "keys") else row["jwt_token"]
            return str(value) if value else None

    def update_node_jwt(self, node_id: str, jwt_token: str) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE active_nodes
                SET jwt_token = ?, session_token = COALESCE(session_token, ?), updated_ts = ?
                WHERE node_id = ?
                """,
                (jwt_token, jwt_token, int(time.time()), node_id),
            )
            return result.rowcount > 0

    def update_heartbeat(self, node_id: str, *args: Any, **kwargs: Any) -> None:
        if args and isinstance(args[0], str):
            status = args[0]
            ts = int(args[1]) if len(args) > 1 else int(kwargs.get("ts", time.time()))
            task_id = args[2] if len(args) > 2 else kwargs.get("task_id")
            seq = kwargs.get("seq")
            gpu_util = kwargs.get("gpu_util")
            vram_free_mb = kwargs.get("vram_free_mb")
        else:
            seq = kwargs.get("seq", args[0] if len(args) > 0 else None)
            gpu_util = kwargs.get("gpu_util", args[1] if len(args) > 1 else None)
            vram_free_mb = kwargs.get("vram_free_mb", args[2] if len(args) > 2 else None)
            task_id = kwargs.get("current_task", args[3] if len(args) > 3 else None)
            ts = int(kwargs.get("ts", time.time()))
            status = "working" if task_id else "idle"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE active_nodes
                SET last_heartbeat_ts = ?,
                    status = ?,
                    current_task_id = ?,
                    heartbeat_seq = COALESCE(?, heartbeat_seq),
                    gpu_util = COALESCE(?, gpu_util),
                    vram_free_mb = COALESCE(?, vram_free_mb),
                    updated_ts = ?
                WHERE node_id = ?
                """,
                (ts, status, task_id, seq, gpu_util, vram_free_mb, ts, node_id),
            )

    def set_node_contributing(self, node_id: str, contributing: bool) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE active_nodes
                SET contributing = ?, updated_ts = ?
                WHERE node_id = ?
                """,
                (1 if contributing else 0, int(time.time()), node_id),
            )
            return result.rowcount > 0

    def update_node_benchmark(self, node_id: str, benchmark_json: str) -> bool:
        with self._connect() as conn:
            result = conn.execute(
                """
                UPDATE active_nodes
                SET benchmark_json = ?, updated_ts = ?
                WHERE node_id = ?
                """,
                (benchmark_json, int(time.time()), node_id),
            )
            return result.rowcount > 0

    def get_all_nodes(self) -> list[dict[str, Any]]:
        now = int(time.time())
        with self._conn_lock:
            conn = self._ensure_conn()
            cur = conn.execute(
                """
            SELECT node_id, session_token, hardware_json, benchmark_json, status,
                   current_task_id, contributing, jobs_completed, jobs_failed,
                   reliability_score, last_heartbeat_ts
            FROM active_nodes
            ORDER BY node_id ASC
            """
            )
            rows = cur.fetchall()
        nodes = []
        for row in rows:
            hardware = self._safe_json(row["hardware_json"])
            benchmark = self._safe_json(row["benchmark_json"])
            last_heartbeat_ts = row["last_heartbeat_ts"]
            nodes.append(
                {
                    "node_id": row["node_id"],
                    "status": row["status"],
                    "session_token_present": bool(row["session_token"]),
                    "hardware": hardware,
                    "benchmark": benchmark,
                    "current_task_id": row["current_task_id"],
                    "contributing": bool(row["contributing"]),
                    "jobs_completed": row["jobs_completed"] or 0,
                    "jobs_failed": row["jobs_failed"] or 0,
                    "reliability_score": row["reliability_score"] or 1.0,
                    "last_heartbeat_ts": last_heartbeat_ts,
                    "heartbeat_age_seconds": None
                    if last_heartbeat_ts is None
                    else max(0, now - last_heartbeat_ts),
                }
            )
        return nodes

    def list_trusted_submitters(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT node_id, created_ts FROM trusted_submitters ORDER BY created_ts DESC"
            ).fetchall()
            return [{"node_id": r["node_id"], "created_ts": int(r["created_ts"])} for r in rows]

    def add_trusted_submitter(self, node_id: str) -> None:
        if not re.match(r"^[a-zA-Z0-9_-]{1,128}$", node_id):
            raise ValueError("invalid node_id")
        ts = int(time.time())
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO trusted_submitters (node_id, created_ts) VALUES (?, ?)",
                (node_id, ts),
            )

    def remove_trusted_submitter(self, node_id: str) -> bool:
        if not re.match(r"^[a-zA-Z0-9_-]{1,128}$", node_id):
            raise ValueError("invalid node_id")
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM trusted_submitters WHERE node_id = ?", (node_id,))
            return cur.rowcount > 0

