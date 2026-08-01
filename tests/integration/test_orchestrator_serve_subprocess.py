"""Integration: orchestrator ``serve()`` entrypoint in a real subprocess (CI harness)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SERVE_ADMIN_PORT = 19880
_SERVE_GRPC_PORT = 19881


def _wait_health(admin_port: int, timeout_s: float = 25.0) -> dict:
    url = f"http://127.0.0.1:{admin_port}/admin/health"
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                body = json.loads(resp.read().decode())
                if body.get("ok"):
                    return body
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = exc
        time.sleep(0.4)
    raise AssertionError(f"orchestrator admin health not ready: {last_error}")


@pytest.mark.integration
@pytest.mark.slow
def test_orchestrator_serve_subprocess_exposes_admin_health(tmp_path):
    """``python -m services_python.orchestrator_grpc`` binds admin HTTP for smoke/CI."""
    env = os.environ.copy()
    env["ADMIN_HOST"] = "127.0.0.1"
    env["ADMIN_PORT"] = str(_SERVE_ADMIN_PORT)
    env["GRPC_PORT"] = str(_SERVE_GRPC_PORT)
    env.pop("ADMIN_REQUIRE_AUTH", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "services_python.orchestrator_grpc"],
        cwd=str(_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        body = _wait_health(_SERVE_ADMIN_PORT)
        assert "job_submission_available" in body
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
