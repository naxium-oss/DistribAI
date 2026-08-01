"""Penetration checks against live ``python -m services_python.orchestrator_grpc`` subprocess."""

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

from tests.fast_env import poll_seconds

_ROOT = Path(__file__).resolve().parents[2]
_ADMIN_PORT = 19882
_GRPC_PORT = 19883
_ADMIN = f"http://127.0.0.1:{_ADMIN_PORT}"
_SECRET = "pen-test-admin-secret"


def _wait_health(timeout_s: float = 20.0) -> dict:
    url = f"{_ADMIN}/admin/health"
    deadline = time.time() + timeout_s
    poll = poll_seconds(0.4)
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                body = json.loads(resp.read().decode())
                if body.get("ok"):
                    return body
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = exc
        time.sleep(poll)
    raise AssertionError(f"orchestrator admin health not ready: {last_error}")


def _request(method: str, path: str, *, token: str | None = None, body: dict | None = None) -> int:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{_ADMIN}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


@pytest.fixture(scope="module")
def orchestrator_proc():
    env = os.environ.copy()
    env["ADMIN_HOST"] = "127.0.0.1"
    env["ADMIN_PORT"] = str(_ADMIN_PORT)
    env["GRPC_PORT"] = str(_GRPC_PORT)
    env["ADMIN_REQUIRE_AUTH"] = "1"
    env["DISTRIBAI_ADMIN_SECRET"] = _SECRET
    env["JWT_SECRET"] = "pen-test-jwt-secret-32chars-min"
    env["SIGNING_KEY"] = "pen-test-signing-key-32chars-min"
    proc = subprocess.Popen(
        [sys.executable, "-m", "services_python.orchestrator_grpc"],
        cwd=str(_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_health()
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.integration
@pytest.mark.security
def test_live_serve_blocks_unauthenticated_admin_jobs(orchestrator_proc):
    assert _request("GET", "/admin/jobs") == 401
    assert _request("POST", "/admin/jobs", body={"model_name": "m", "steps": 1}) == 401


@pytest.mark.integration
@pytest.mark.security
def test_live_serve_blocks_unauthenticated_admin_stream(orchestrator_proc):
    assert _request("GET", "/admin/stream") == 401


@pytest.mark.integration
@pytest.mark.security
def test_live_serve_blocks_unauthenticated_command_triggers(orchestrator_proc):
    assert _request("POST", "/api/admin/distribai/registry/sync", body={}) == 401
    assert _request("POST", "/api/admin/public-release/publish", body={}) == 401


@pytest.mark.integration
@pytest.mark.security
def test_live_serve_allows_bearer_on_protected_routes(orchestrator_proc):
    assert _request("GET", "/admin/jobs", token=_SECRET) == 200
    assert _request("GET", "/admin/nodes", token=_SECRET) == 200
