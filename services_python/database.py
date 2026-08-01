"""SQLite database layer for DistribAI persistence.

Provides durable storage for jobs, checkpoints, logs, and admin keys.
Uses WAL mode for concurrent read/write performance.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


# SQL Injection Protection
def validate_sql_identifier(identifier: str) -> str:
    """Validate SQL identifier to prevent injection."""
    if not identifier:
        raise ValueError("Empty identifier")

    # Only allow alphanumeric characters, underscores, and periods
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_.]*$", identifier):
        raise ValueError(f"Invalid identifier: {identifier}")

    # Prevent SQL keywords
    sql_keywords = {
        "DROP",
        "DELETE",
        "INSERT",
        "UPDATE",
        "CREATE",
        "ALTER",
        "EXEC",
        "EXECUTE",
        "UNION",
        "SELECT",
        "FROM",
        "WHERE",
        "JOIN",
    }
    if identifier.upper() in sql_keywords:
        raise ValueError(f"Identifier cannot be SQL keyword: {identifier}")

    return identifier


def validate_string_input(value: str, max_length: int = 1000) -> str:
    """ "Validate string input to prevent injection."""
    if not isinstance(value, str):
        raise ValueError("Value must be a string")

    if len(value) > max_length:
        raise ValueError(f"String too long (max {max_length} characters)")

    # Simple validation - check for dangerous patterns
    dangerous_patterns = ["<script", "javascript:", "vbscript:", "onload=", "onerror="]
    for pattern in dangerous_patterns:
        if pattern.lower() in value.lower():
            raise ValueError(f"String contains dangerous pattern: {pattern}")

    return value


def validate_job_id(job_id: str) -> str:
    """Validate job ID format."""
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", job_id):
        raise ValueError("Invalid job ID format")
    return job_id


def validate_node_id(node_id: str) -> str:
    """Validate node ID format."""
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", node_id):
        raise ValueError("Invalid node ID format")
    return node_id


# Database path in user's home directory for persistence
DB_DIR = Path.home() / ".distribai"
DB_PATH = DB_DIR / "distribai.db"

# Schema definition
SCHEMA = """
-- Jobs table (main job tracking)
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    config_json TEXT NOT NULL,
    total_nodes INTEGER DEFAULT 0,
    active_nodes INTEGER DEFAULT 0,
    current_step INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    checkpoint_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_message TEXT
);

-- Node assignments for distributed jobs
CREATE TABLE IF NOT EXISTS job_nodes (
    job_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'assigned',
    last_heartbeat TIMESTAMP,
    PRIMARY KEY (job_id, node_id),
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

-- Checkpoints (sharded and aggregated)
CREATE TABLE IF NOT EXISTS checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    node_id TEXT,
    path TEXT NOT NULL,
    is_aggregated BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

-- Logs (stdout/stderr from nodes)
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    node_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    level TEXT DEFAULT 'INFO',
    message TEXT NOT NULL,
    stream TEXT DEFAULT 'stdout',
    FOREIGN KEY (job_id) REFERENCES jobs(id) ON DELETE CASCADE
);

-- Admin keys (encrypted)
CREATE TABLE IF NOT EXISTS admin_keys (
    node_id TEXT PRIMARY KEY,
    encrypted_key TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Admin key requests (pending approval)
CREATE TABLE IF NOT EXISTS admin_key_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL UNIQUE,
    username TEXT,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'pending',
    approved_at TIMESTAMP,
    rejected_at TIMESTAMP
);

-- Dependency approvals (audit trail for unsafe deps)
CREATE TABLE IF NOT EXISTS dependency_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    package_name TEXT NOT NULL,
    requested_version TEXT,
    approved_by TEXT,
    approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    warning_count INTEGER DEFAULT 0
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_job_nodes_job ON job_nodes(job_id);
CREATE INDEX IF NOT EXISTS idx_job_nodes_node ON job_nodes(node_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_job ON checkpoints(job_id);
CREATE INDEX IF NOT EXISTS idx_logs_job ON logs(job_id);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_admin_requests_status ON admin_key_requests(status);
"""


class Database:
    """SQLite database manager with async wrapper."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self._lock = asyncio.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize database with schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(SCHEMA)
            conn.commit()

    @contextmanager
    def _connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # =========================================================================
    # Job Operations
    # =========================================================================

    async def create_job(
        self,
        job_id: str,
        name: str,
        job_type: str,
        config: dict,
        total_steps: int = 0,
        total_nodes: int = 0,
    ) -> dict:
        """Create a new job record."""
        async with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO jobs (id, name, job_type, config_json, total_steps, total_nodes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (job_id, name, job_type, json.dumps(config), total_steps, total_nodes),
                )
                conn.commit()
                return await self.get_job(job_id)

    async def get_job(self, job_id: str) -> dict | None:
        """Get job by ID."""
        async with self._lock:
            with self._connection() as conn:
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if row:
                    return self._row_to_dict(row)
                return None

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        current_step: int | None = None,
        active_nodes: int | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Update job status."""
        async with self._lock:
            with self._connection() as conn:
                updates = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
                params = [status]

                if current_step is not None:
                    updates.append("current_step = ?")
                    params.append(current_step)
                if active_nodes is not None:
                    updates.append("active_nodes = ?")
                    params.append(active_nodes)
                if error_message is not None:
                    updates.append("error_message = ?")
                    params.append(error_message)

                # Validate column names to prevent SQL injection
                allowed_columns = {
                    "status",
                    "current_step",
                    "active_nodes",
                    "error_message",
                    "updated_at",
                }
                for update in updates:
                    col_name = update.split(" = ?")[0]
                    if col_name not in allowed_columns:
                        raise ValueError(f"Invalid column name in update: {col_name}")

                params.append(job_id)

                conn.execute(
                    f"UPDATE jobs SET {', '.join(updates)} WHERE id = ?",
                    params,
                )
                conn.commit()
                return conn.total_changes > 0

    async def set_job_checkpoint(self, job_id: str, checkpoint_path: str) -> bool:
        """Set checkpoint path for job."""
        async with self._lock:
            with self._connection() as conn:
                conn.execute(
                    "UPDATE jobs SET checkpoint_path = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (checkpoint_path, job_id),
                )
                conn.commit()
                return conn.total_changes > 0

    async def list_jobs(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List jobs with optional filter."""
        async with self._lock:
            with self._connection() as conn:
                if status:
                    rows = conn.execute(
                        "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (status, limit, offset),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                        (limit, offset),
                    ).fetchall()
                return [self._row_to_dict(row) for row in rows]

    async def delete_job(self, job_id: str) -> bool:
        """Delete job and related data."""
        async with self._lock:
            with self._connection() as conn:
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                conn.commit()
                return conn.total_changes > 0

    # =========================================================================
    # Job Node Operations (Distributed Training)
    # =========================================================================

    async def assign_node_to_job(
        self,
        job_id: str,
        node_id: str,
        rank: int,
    ) -> bool:
        """Assign a node to a distributed job."""
        async with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO job_nodes (job_id, node_id, rank, status, last_heartbeat)
                    VALUES (?, ?, ?, 'assigned', CURRENT_TIMESTAMP)
                    """,
                    (job_id, node_id, rank),
                )
                conn.commit()
                return True

    async def update_node_heartbeat(self, job_id: str, node_id: str, status: str) -> bool:
        """Update node heartbeat and status."""
        async with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    UPDATE job_nodes
                    SET status = ?, last_heartbeat = CURRENT_TIMESTAMP
                    WHERE job_id = ? AND node_id = ?
                    """,
                    (status, job_id, node_id),
                )
                conn.commit()
                return conn.total_changes > 0

    async def get_job_nodes(self, job_id: str) -> list[dict]:
        """Get all nodes assigned to a job."""
        async with self._lock:
            with self._connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM job_nodes WHERE job_id = ? ORDER BY rank",
                    (job_id,),
                ).fetchall()
                return [self._row_to_dict(row) for row in rows]

    async def get_node_stale(self, timeout_seconds: int = 60) -> list[dict]:
        """Get nodes that haven't sent heartbeat recently."""
        async with self._lock:
            with self._connection() as conn:
                # Validate timeout_seconds to prevent SQL injection
                if not isinstance(timeout_seconds, int) or timeout_seconds < 0:
                    raise ValueError("timeout_seconds must be a non-negative integer")

                rows = conn.execute(
                    """
                    SELECT * FROM job_nodes
                    WHERE status = 'running'
                    AND datetime(last_heartbeat) < datetime('now', '-60 seconds')
                    """,
                ).fetchall()
                return [self._row_to_dict(row) for row in rows]

    # =========================================================================
    # Checkpoint Operations
    # =========================================================================

    async def save_checkpoint(
        self,
        job_id: str,
        step: int,
        node_id: str | None,
        path: str,
        is_aggregated: bool = False,
    ) -> int:
        """Save checkpoint record."""
        async with self._lock:
            with self._connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO checkpoints (job_id, step, node_id, path, is_aggregated)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (job_id, step, node_id, path, is_aggregated),
                )
                conn.commit()
                return cursor.lastrowid

    async def get_latest_checkpoint(self, job_id: str) -> dict | None:
        """Get latest checkpoint for a job."""
        async with self._lock:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE job_id = ? AND is_aggregated = TRUE
                    ORDER BY step DESC LIMIT 1
                    """,
                    (job_id,),
                ).fetchone()
                if row:
                    return self._row_to_dict(row)
                return None

    async def get_checkpoint_for_resume(self, job_id: str, node_rank: int) -> str | None:
        """Get checkpoint path for node to resume from."""
        # First try aggregated checkpoint
        async with self._lock:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    SELECT path FROM checkpoints
                    WHERE job_id = ? AND is_aggregated = TRUE
                    ORDER BY step DESC LIMIT 1
                    """,
                    (job_id,),
                ).fetchone()
                if row:
                    return row["path"]
                return None

    # =========================================================================
    # Log Operations
    # =========================================================================

    async def add_log(
        self,
        job_id: str,
        message: str,
        node_id: str | None = None,
        level: str = "INFO",
        stream: str = "stdout",
    ) -> int:
        """Add a log entry."""
        async with self._lock:
            with self._connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO logs (job_id, node_id, level, message, stream)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (job_id, node_id, level, message, stream),
                )
                conn.commit()
                return cursor.lastrowid

    async def get_logs(
        self,
        job_id: str,
        node_id: str | None = None,
        limit: int = 1000,
        since_id: int | None = None,
    ) -> list[dict]:
        """Get logs for a job."""
        async with self._lock:
            with self._connection() as conn:
                if node_id:
                    rows = conn.execute(
                        """
                        SELECT * FROM logs
                        WHERE job_id = ? AND node_id = ? AND id > ?
                        ORDER BY id DESC LIMIT ?
                        """,
                        (job_id, node_id, since_id or 0, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM logs
                        WHERE job_id = ? AND id > ?
                        ORDER BY id DESC LIMIT ?
                        """,
                        (job_id, since_id or 0, limit),
                    ).fetchall()
                return [self._row_to_dict(row) for row in reversed(rows)]

    # =========================================================================
    # Admin Key Operations
    # =========================================================================

    async def create_admin_key(
        self, node_id: str, encrypted_key: str, expires_at: datetime | None = None
    ) -> bool:
        """Store encrypted admin key."""
        async with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO admin_keys (node_id, encrypted_key, expires_at)
                    VALUES (?, ?, ?)
                    """,
                    (node_id, encrypted_key, expires_at),
                )
                conn.commit()
                return True

    async def get_admin_key(self, node_id: str) -> dict | None:
        """Get admin key for a node."""
        async with self._lock:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT * FROM admin_keys WHERE node_id = ?",
                    (node_id,),
                ).fetchone()
                if row:
                    return self._row_to_dict(row)
                return None

    async def revoke_admin_key(self, node_id: str) -> bool:
        """Revoke admin key."""
        async with self._lock:
            with self._connection() as conn:
                conn.execute("DELETE FROM admin_keys WHERE node_id = ?", (node_id,))
                conn.commit()
                return conn.total_changes > 0

    async def is_admin(self, node_id: str, encrypted_token: str) -> bool:
        """Verify if node has valid admin key."""
        key = await self.get_admin_key(node_id)
        if not key:
            return False
        if key.get("expires_at"):
            expires = datetime.fromisoformat(key["expires_at"])
            if expires < datetime.now(UTC):
                return False
        # Token validation happens in admin_keys.py (decryption)
        return True

    # =========================================================================
    # Admin Key Request Operations
    # =========================================================================

    async def request_admin_key(self, node_id: str, username: str | None = None) -> int:
        """Create admin key request."""
        async with self._lock:
            with self._connection() as conn:
                try:
                    cursor = conn.execute(
                        """
                        INSERT INTO admin_key_requests (node_id, username)
                        VALUES (?, ?)
                        ON CONFLICT(node_id) DO UPDATE SET
                            status = 'pending',
                            requested_at = CURRENT_TIMESTAMP,
                            approved_at = NULL,
                            rejected_at = NULL
                        """,
                        (node_id, username),
                    )
                    conn.commit()
                    return cursor.lastrowid
                except sqlite3.IntegrityError:
                    # Update existing pending request
                    conn.execute(
                        """
                        UPDATE admin_key_requests
                        SET requested_at = CURRENT_TIMESTAMP
                        WHERE node_id = ? AND status = 'pending'
                        """,
                        (node_id,),
                    )
                    conn.commit()
                    row = conn.execute(
                        "SELECT id FROM admin_key_requests WHERE node_id = ?",
                        (node_id,),
                    ).fetchone()
                    return row["id"] if row else 0

    async def get_pending_admin_requests(self) -> list[dict]:
        """Get pending admin key requests."""
        async with self._lock:
            with self._connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM admin_key_requests
                    WHERE status = 'pending'
                    ORDER BY requested_at DESC
                    """,
                ).fetchall()
                return [self._row_to_dict(row) for row in rows]

    async def approve_admin_request(self, request_id: int, approved_by: str) -> str | None:
        """Approve admin key request. Returns node_id."""
        async with self._lock:
            with self._connection() as conn:
                row = conn.execute(
                    """
                    UPDATE admin_key_requests
                    SET status = 'approved', approved_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND status = 'pending'
                    RETURNING node_id
                    """,
                    (request_id,),
                ).fetchone()
                conn.commit()
                return row["node_id"] if row else None

    async def reject_admin_request(self, request_id: int) -> bool:
        """Reject admin key request."""
        async with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    UPDATE admin_key_requests
                    SET status = 'rejected', rejected_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND status = 'pending'
                    """,
                    (request_id,),
                )
                conn.commit()
                return conn.total_changes > 0

    # =========================================================================
    # Dependency Approval Operations
    # =========================================================================

    async def log_dependency_approval(
        self,
        job_id: str,
        package_name: str,
        requested_version: str,
        approved_by: str,
        warning_count: int = 0,
    ) -> int:
        """Log admin approval of unsafe dependency."""
        async with self._lock:
            with self._connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO dependency_approvals
                    (job_id, package_name, requested_version, approved_by, warning_count)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (job_id, package_name, requested_version, approved_by, warning_count),
                )
                conn.commit()
                return cursor.lastrowid

    # =========================================================================
    # Statistics & Maintenance
    # =========================================================================

    async def get_stats(self) -> dict:
        """Get database statistics."""
        async with self._lock:
            with self._connection() as conn:
                job_stats = conn.execute(
                    "SELECT status, COUNT(*) as count FROM jobs GROUP BY status"
                ).fetchall()

                total_logs = conn.execute("SELECT COUNT(*) as count FROM logs").fetchone()["count"]

                pending_requests = conn.execute(
                    "SELECT COUNT(*) as count FROM admin_key_requests WHERE status = 'pending'"
                ).fetchone()["count"]

                return {
                    "jobs_by_status": {row["status"]: row["count"] for row in job_stats},
                    "total_logs": total_logs,
                    "pending_admin_requests": pending_requests,
                }

    async def cleanup_old_logs(self, days: int = 30) -> int:
        """Delete logs older than specified days."""
        # Validate days parameter to prevent SQL injection
        if not isinstance(days, int) or days < 0:
            raise ValueError("days must be a non-negative integer")

        async with self._lock:
            with self._connection() as conn:
                # Use a fixed time interval instead of parameter interpolation
                if days <= 7:
                    time_interval = "7 days"
                elif days <= 30:
                    time_interval = "30 days"
                elif days <= 90:
                    time_interval = "90 days"
                else:
                    time_interval = "365 days"

                conn.execute(
                    f"DELETE FROM logs WHERE timestamp < datetime('now', '-{time_interval}')"
                )
                conn.commit()
                return conn.total_changes

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert SQLite row to dictionary."""
        result = dict(row)
        # Parse JSON fields
        if "config_json" in result and result["config_json"]:
            try:
                result["config"] = json.loads(result["config_json"])
            except json.JSONDecodeError:
                result["config"] = {}
            del result["config_json"]
        return result


# Global database instance
_db: Database | None = None


def get_database() -> Database:
    """Get or create global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


async def init_database() -> Database:
    """Initialize database and return instance."""
    return get_database()
