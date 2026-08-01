"""Shared HTTP client for the orchestrator admin API.

Used by both the flat CLI (``distribai_cli.py``) and the TUI (``tui.py``) so
the two surfaces never drift on auth headers, timeouts, or error shapes.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


class AdminAPIError(RuntimeError):
    """Raised when the admin API returns an error payload or HTTP failure."""


class AdminAPIClient:
    """Thin ``urllib``-based client for ``/admin/*`` and ``/jobs/*`` routes.

    No third-party HTTP dependency is required: the admin API is polled
    infrequently enough (CLI commands, TUI refresh ticks) that ``urllib``
    is simpler than adding ``requests``/``httpx`` as a hard dependency.
    """

    def __init__(self, base_url: str | None = None, *, timeout: float = 30.0) -> None:
        self.base_url = (base_url or os.getenv("ORCHESTRATOR_ADMIN_URL", "http://127.0.0.1:8766")).rstrip(
            "/"
        )
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        secret = os.getenv("DISTRIBAI_ADMIN_SECRET", "").strip()
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        return headers

    def request(
        self, path: str, method: str = "GET", payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Issue one request; always returns a dict (``{"error": ...}`` on failure)."""
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                detail = {}
            return {"error": detail.get("error") or f"HTTP {exc.code}: {exc.reason}"}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {"error": str(exc)}

    def get(self, path: str) -> dict[str, Any]:
        return self.request(path, "GET")

    def post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request(path, "POST", payload)

    def delete(self, path: str) -> dict[str, Any]:
        return self.request(path, "DELETE")

    def health(self) -> dict[str, Any]:
        return self.get("/admin/health")

    def is_reachable(self) -> bool:
        result = self.health()
        return "error" not in result

    def list_nodes(self) -> list[dict[str, Any]]:
        result = self.get("/admin/nodes")
        nodes = result.get("nodes")
        return nodes if isinstance(nodes, list) else []

    def list_jobs(self) -> list[dict[str, Any]]:
        result = self.get("/admin/jobs")
        jobs = result.get("jobs")
        return jobs if isinstance(jobs, list) else []

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self.get(f"/admin/jobs/{job_id}")

    def list_credits(self) -> dict[str, Any]:
        result = self.get("/admin/credits")
        credits = result.get("credits")
        return credits if isinstance(credits, dict) else {}

    def tail_logs(self, n: int = 100) -> list[str]:
        result = self.get(f"/admin/logs?n={int(n)}")
        logs = result.get("logs")
        return logs if isinstance(logs, list) else []
