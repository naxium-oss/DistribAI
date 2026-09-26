"""Connection, schema, and shared DBManager helpers."""

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)
RETRYABLE_TASK_STATUSES = {"error", "failed", "timeout", "invalid_gradient"}


class DBManagerBase:
    """SQLite database manager base: connection, schema, migrations."""

    def __init__(self, db_path: str, schema_path: str) -> None:
        self.db_path = db_path
        self._conn_lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._init_schema(schema_path)
        self._apply_migrations()

    def _create_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=60.0,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 60000")
        if self.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._create_conn()
        return self._conn

    @contextmanager
    def _connect(self):
        """Serialize SQLite access across threads (asyncio default executor)."""
        with self._conn_lock:
            conn = self._ensure_conn()
            with conn:
                yield conn

    def _init_schema(self, schema_path: str) -> None:
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema not found at {schema_path}")
        with open(schema_path, encoding="utf-8") as f:
            script = f.read()
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(script)

    def _apply_migrations(self) -> None:
        with self._connect() as conn:
            self._ensure_column(conn, "active_nodes", "jwt_token", "TEXT")
            self._ensure_column(conn, "active_nodes", "benchmark_json", "TEXT")
            self._ensure_column(conn, "active_nodes", "contributing", "INTEGER DEFAULT 1")
            self._ensure_column(conn, "active_nodes", "current_task_id", "TEXT")
            self._ensure_column(conn, "active_nodes", "jobs_completed", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "active_nodes", "jobs_failed", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "active_nodes", "reliability_score", "REAL DEFAULT 1.0")
            self._ensure_column(conn, "active_nodes", "created_ts", "INTEGER")
            self._ensure_column(conn, "active_nodes", "updated_ts", "INTEGER")
            self._ensure_column(conn, "active_nodes", "heartbeat_seq", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "active_nodes", "gpu_util", "REAL")
            self._ensure_column(conn, "active_nodes", "vram_free_mb", "INTEGER")
            self._ensure_column(conn, "jobs", "job_type", "TEXT DEFAULT 'fine_tune'")
            self._ensure_column(conn, "jobs", "base_model", "TEXT")
            self._ensure_column(conn, "jobs", "dataset_ref", "TEXT")
            self._ensure_column(conn, "jobs", "description", "TEXT")
            self._ensure_column(conn, "jobs", "priority_tier", "TEXT DEFAULT 'P1'")
            self._ensure_column(conn, "jobs", "total_votes", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "jobs", "vote_weight", "REAL DEFAULT 1.0")
            self._ensure_column(conn, "jobs", "submitter_id", "TEXT DEFAULT 'distribai'")
            self._ensure_column(conn, "jobs", "org", "TEXT DEFAULT 'DistribAI'")
            self._ensure_column(conn, "jobs", "updated_ts", "INTEGER")
            self._ensure_column(conn, "jobs", "started_ts", "INTEGER")
            self._ensure_column(conn, "jobs", "completed_ts", "INTEGER")
            self._ensure_column(conn, "jobs", "steps", "INTEGER DEFAULT 100")
            self._ensure_column(conn, "jobs", "batch_size", "INTEGER DEFAULT 32")
            self._ensure_column(conn, "jobs", "queue_position", "INTEGER")
            self._ensure_column(conn, "jobs", "estimated_start_hours", "REAL")
            self._ensure_column(conn, "jobs", "active_nodes", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "jobs", "progress_pct", "REAL DEFAULT 0")
            self._ensure_column(conn, "jobs", "current_step", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "jobs", "total_steps", "INTEGER DEFAULT 100")
            self._ensure_column(conn, "jobs", "attempts", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "jobs", "latest_task_id", "TEXT")
            self._ensure_column(conn, "jobs", "latest_reason", "TEXT")
            self._ensure_column(conn, "jobs", "aggregate_json", "TEXT")
            self._ensure_column(conn, "tasks", "weight_version", "TEXT")
            self._ensure_column(conn, "tasks", "steps", "INTEGER DEFAULT 1")
            self._ensure_column(conn, "tasks", "step_offset", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "tasks", "attempt_count", "INTEGER DEFAULT 0")
            self._ensure_column(conn, "tasks", "max_attempts", "INTEGER DEFAULT 3")
            self._ensure_column(conn, "tasks", "benchmark_score", "REAL")
            self._ensure_column(conn, "tasks", "reliability_score", "REAL")
            self._ensure_column(conn, "tasks", "created_ts", "INTEGER")
            self._ensure_column(conn, "tasks", "updated_ts", "INTEGER")
            self._ensure_column(conn, "tasks", "started_ts", "INTEGER")
            self._ensure_column(conn, "tasks", "completed_ts", "INTEGER")
            self._ensure_column(conn, "tasks", "gradient_blob_url", "TEXT")
            self._ensure_column(conn, "tasks", "output_json", "TEXT")
            self._ensure_column(conn, "tasks", "last_error", "TEXT")
            # Structured metadata for credit rows (transfer_id, memo, job_id).
            # Previously transfers hijacked tx_hash/prev_hash for this, which
            # broke the ledger hash chain; those columns now hold a real chain.
            self._ensure_column(conn, "credit_ledger", "metadata_json", "TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vote_transactions (
                    vote_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    voter_id TEXT NOT NULL,
                    credits INTEGER NOT NULL,
                    created_ts INTEGER NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_queue ON tasks(status, deadline_ts, created_ts)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_job_id ON tasks(job_id, status)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_votes_job_id ON vote_transactions(job_id, created_ts)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trusted_submitters (
                    node_id TEXT PRIMARY KEY,
                    created_ts INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trusted_submitters_ts ON trusted_submitters(created_ts)"
            )
            now = int(time.time())
            conn.execute(
                """
                UPDATE active_nodes
                SET created_ts = COALESCE(created_ts, ?),
                    updated_ts = COALESCE(updated_ts, ?)
                """,
                (now, now),
            )
            conn.execute(
                """
                UPDATE jobs
                SET updated_ts = COALESCE(updated_ts, created_ts, ?),
                    total_steps = COALESCE(total_steps, steps, 100)
                """,
                (now,),
            )
            conn.execute(
                """
                UPDATE tasks
                SET created_ts = COALESCE(created_ts, ?),
                    updated_ts = COALESCE(updated_ts, created_ts, ?)
                """,
                (now, now),
            )

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
        if not table.isidentifier() or not column.isidentifier():
            logger.warning("[SECURITY] Invalid table or column name: %s.%s", table, column)
            return
        allowed_types = {
            "TEXT",
            "INTEGER",
            "REAL",
            "BLOB",
            "NUMERIC",
            "BOOLEAN",
            "DATETIME",
            "FLOAT",
            "DOUBLE",
            "INT",
            "BIGINT",
            "SMALLINT",
            "TINYINT",
            "VARCHAR",
            "CHAR",
            "NVARCHAR",
            "STRING",
        }
        ddl_upper = ddl.upper().strip()
        base_type = ddl_upper.split("(")[0].split()[0].strip()
        if not base_type.replace("_", "").isalnum():
            logger.warning("[SECURITY] Invalid characters in DDL type: %s", ddl)
            return
        if base_type not in allowed_types:
            logger.warning("[SECURITY] Invalid DDL type: %s (from %s)", base_type, ddl)
            return
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    def _safe_json(self, value: Any) -> Any:
        if value in (None, ""):
            return {}
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
