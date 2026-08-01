from __future__ import annotations

import asyncio
import os
import socket
import sys
import tempfile
import time

import pytest

os.environ["JWT_SECRET"] = "test_secret_32_bytes_long_for_jwt_signing"
os.environ["S3_BUCKET_NAME"] = "test-bucket"
os.environ["RATE_LIMIT_DISABLED"] = "1"
if "AWS_ACCESS_KEY_ID" not in os.environ:
    os.environ["AWS_ACCESS_KEY_ID"] = "mock_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "mock_secret"
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_root, "services_python"))
sys.path.insert(0, os.path.join(_root, "worker", "src"))
sys.path.insert(0, os.path.join(_root, "worker", "src", "distribai_proto"))

from daemon.daemon import WorkerDaemon
from orchestrator_grpc import serve as start_orchestrator

from tests.fast_env import poll_seconds, startup_seconds


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_registration_and_heartbeat():
    """
    Spins up the orchestrator and daemon locally and verifies the gRPC handshake.
    """
    os.environ["GRPC_PORT"] = str(_free_port())
    os.environ["ADMIN_PORT"] = str(_free_port())
    grpc_url = f"127.0.0.1:{os.environ['GRPC_PORT']}"
    runtime = await start_orchestrator()
    await asyncio.sleep(startup_seconds(2))
    daemon = WorkerDaemon(
        orchestrator_url=grpc_url,
        node_id="test-node-001",
        state_dir=tempfile.mkdtemp(prefix="distribai-integration-"),
    )
    daemon_task = asyncio.create_task(daemon.run())
    timeout = 15
    start = time.time()
    poll = poll_seconds(0.5)
    while not daemon.connected and time.time() - start < timeout:
        await asyncio.sleep(poll)
    assert daemon.connected, "Daemon failed to connect to orchestrator"
    print("Successfully verified Registration Handshake")
    fast = os.getenv("DISTRIBAI_FAST_TEST", "1").strip().lower() not in ("0", "false", "no", "off")
    hb_deadline = time.time() + (5.0 if fast else 12.0)
    while daemon._seq <= 0 and time.time() < hb_deadline:
        await asyncio.sleep(poll)
    assert daemon._seq > 0, "No heartbeats sent after registration"
    print(f"Verified Heartbeat Seq: {daemon._seq}")
    daemon.stop()
    daemon_task.cancel()
    try:
        await asyncio.wait_for(daemon_task, timeout=5)
    except (asyncio.CancelledError, TimeoutError):
        pass
    try:
        await runtime.stop()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(test_registration_and_heartbeat())
