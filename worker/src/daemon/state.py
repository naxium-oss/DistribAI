"""
WorkerState — persists worker runtime state to disk so the dashboard
and other tools can observe it without a live connection to the daemon.
"""

import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuthTokens:
    """Authentication tokens for worker node."""

    jwt_token: str | None = None
    session_token: str | None = None
    node_id: str | None = None
    expires_at: float | None = None


class WorkerState:
    """
    Writes {state_dir}/{node_id}/state.json (current snapshot)
    and appends to {state_dir}/{node_id}/events.jsonl (rolling event log).
    Atomic writes via rename so readers never see a partial file.
    """

    def __init__(self, state_dir: str, node_id: str, *, ephemeral: bool = False) -> None:
        self.node_dir = Path(state_dir) / node_id
        self.node_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.node_dir / "state.json"
        self.events_file = self.node_dir / "events.jsonl"
        self.auth_file = self.node_dir / "auth.json"
        self.node_id = node_id
        self._ephemeral = ephemeral
        self._memory_auth: AuthTokens | None = None
        self._last_heartbeat_flush_at: float = 0.0
        self._heartbeat_flush_interval: float = max(
            0.5,
            float(os.getenv("WORKER_STATE_HEARTBEAT_FLUSH_SEC", "2.0")),
        )
        self._state: dict[str, Any] = {
            "node_id": node_id,
            "status": "starting",
            "started_at": time.time(),
            "last_heartbeat_at": None,
            "heartbeat_seq": 0,
            "current_job": None,
            "jobs_completed": 0,
            "jobs_failed": 0,
        }
        if self.state_file.exists():
            persisted = self._load_from_file()
            auth = persisted.pop("auth", None)
            self._state.update(persisted)
            if auth is not None:
                self._state["auth"] = auth
        self._flush()

    def set_status(self, status: str, job: dict | None = None) -> None:
        self._state["status"] = status
        if job is not None:
            self._state["current_job"] = {
                "job_id": job.get("job_id"),
                "task_id": job.get("task_id"),
                "model_name": job.get("model_name"),
                "steps": job.get("steps"),
                "batch_size": job.get("batch_size"),
                "started_at": time.time(),
            }
        elif status in ("idle", "disconnected", "stopped"):
            self._state["current_job"] = None
        self._log_event("status_change", {"status": status})
        self._flush()

    def update_heartbeat(self, seq: int) -> None:
        """Update heartbeat counters; flush to disk at most every `_heartbeat_flush_interval`.

        Heartbeats are frequent; syncing state.json on every beat thrashes disk and blocks
        the asyncio loop when many workers share one event loop (e.g. stress tests).
        """
        now = time.time()
        self._state["last_heartbeat_at"] = now
        self._state["heartbeat_seq"] = seq
        if now - self._last_heartbeat_flush_at >= self._heartbeat_flush_interval:
            self._last_heartbeat_flush_at = now
            self._flush()

    def flush(self) -> None:
        """Persist current state immediately (e.g. before shutdown)."""
        self._flush()

    def record_job_done(self, status: str) -> None:
        if status == "success":
            self._state["jobs_completed"] += 1
        else:
            self._state["jobs_failed"] += 1
        self._state["current_job"] = None
        self._flush()

    def load_benchmark_results(self) -> dict | None:
        """
        Load benchmark results from temporary file.

        Returns:
            Benchmark results dictionary or None if not found
        """
        results_file = (
            Path(tempfile.gettempdir()) / f"distribai_benchmark_results_{self.node_id}.json"
        )
        if results_file.exists():
            try:
                with open(results_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to load benchmark results: {e}")
        return None

    def save_auth_tokens(
        self, jwt_token: str | None = None, session_token: str | None = None
    ) -> None:
        """
        Save authentication tokens to persistent storage.

        Args:
            jwt_token: JWT authentication token
            session_token: Session token for WebSocket
        """
        tokens = self.load_auth_tokens()
        if jwt_token:
            tokens.jwt_token = jwt_token
        if session_token:
            tokens.session_token = session_token
        tokens.node_id = self.node_id

        if self._ephemeral:
            self._memory_auth = tokens
            return

        data = self._load_from_file()
        data["auth"] = asdict(tokens)
        self._state.update({key: value for key, value in data.items() if key != "auth"})
        self._save_to_file(data)

    def load_auth_tokens(self) -> AuthTokens:
        """
        Load authentication tokens from persistent storage.

        Returns:
            AuthTokens object with current tokens

        Example:
            >>> tokens = state.load_auth_tokens()
            >>> print(f"JWT: {tokens.jwt_token[:20]}...")
        """
        if self._ephemeral:
            return self._memory_auth or AuthTokens(node_id=self.node_id)
        data = self._load_from_file()
        auth_data = data.get("auth", {})
        return AuthTokens(
            jwt_token=auth_data.get("jwt_token"),
            session_token=auth_data.get("session_token"),
            node_id=auth_data.get("node_id"),
            expires_at=auth_data.get("expires_at"),
        )

    def clear_auth_tokens(self) -> None:
        """
        Clear all authentication tokens from storage.

        Called when authentication fails or tokens expire.

        Example:
            >>> state.clear_auth_tokens()
            >>> print("Tokens cleared")
        """
        if self._ephemeral:
            self._memory_auth = AuthTokens(node_id=self.node_id)
            return
        data = self._load_from_file()
        data["auth"] = {}
        self._save_to_file(data)

    def get_status(self) -> str:
        """
        Get the current worker status.

        Returns:
            Current status string

        Example:
            >>> status = state.get_status()
            >>> print(f"Status: {status}")
        """
        data = self._load_from_file()
        return data.get("status", "unknown")

    def _log_event(self, event_type: str, data: dict) -> None:
        """
        Log an event to the events file.

        Args:
            event_type: Type of event
            data: Event data dictionary
        """
        event = {
            "ts": time.time(),
            "type": event_type,
            "data": data,
        }
        try:
            with open(self.events_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError as e:
            logger.error(f"Failed to write event: {e}")

    def _flush(self) -> None:
        """
        Flush current state to disk atomically.

        Writes to temp file then renames for atomicity.
        """
        self._save_to_file(self._state)

    def _load_from_file(self) -> dict[str, Any]:
        """
        Load state data from JSON file.

        Returns:
            Dictionary with state data
        """
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load state file: {e}")
            return {}

    def _save_to_file(self, data: dict[str, Any]) -> None:
        """
        Save state data to JSON file atomically.

        Args:
            data: Dictionary with state data to save
        """
        last_err: OSError | None = None
        for attempt in range(5):
            tmp = self.state_file.with_name(
                f"{self.state_file.stem}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, self.state_file)
                return
            except OSError as e:
                last_err = e
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                if attempt < 4:
                    time.sleep(0.02 * (attempt + 1))
        logger.error("Failed to save state file after retries: %s", last_err)
