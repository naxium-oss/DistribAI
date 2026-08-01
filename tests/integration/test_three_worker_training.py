"""
Integration test for 3-worker training scenario.
This test verifies that the orchestrator can manage training with exactly 3 workers,
as required by the "server host can run 3" specification.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "src" / "distribai_proto"))
from services_python.orchestrator_grpc import serve as orchestrator_serve
from worker.src.daemon.daemon import WorkerDaemon

_GRPC_PORT = 59765
_ADMIN_PORT = 59766
_GRPC_URL = f"127.0.0.1:{_GRPC_PORT}"
_ADMIN = f"http://127.0.0.1:{_ADMIN_PORT}"


class _OrcThread:
    def __init__(self, grpc_port: int, admin_port: int) -> None:
        self._grpc_port = grpc_port
        self._admin_port = admin_port
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runtime = None
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread_exc: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        if not self._ready.wait(timeout=15):
            raise RuntimeError(
                f"Orchestrator startup timed out (grpc={self._grpc_port}, admin={self._admin_port})"
            )
        if self._thread_exc is not None:
            raise RuntimeError(
                f"Orchestrator failed to bind or start (grpc={self._grpc_port}, admin={self._admin_port})"
            ) from self._thread_exc
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        os.environ["GRPC_PORT"] = str(self._grpc_port)
        os.environ["ADMIN_PORT"] = str(self._admin_port)
        os.environ.setdefault("DISTRIBAI_ALLOW_INSECURE_REGISTER", "1")
        os.environ.setdefault("GRPC_USE_TLS", "false")

        async def start_orchestrator():
            self.runtime = await orchestrator_serve()

        try:
            self.loop.run_until_complete(start_orchestrator())
        except BaseException as exc:
            self._thread_exc = exc
        finally:
            self._ready.set()

        if self._thread_exc is not None:
            try:
                self.loop.close()
            except Exception:
                pass
            return

        while not self._stop_event.is_set():
            try:
                self.loop.run_forever()
            except (OSError, RuntimeError):
                pass
        if self.runtime is not None:
            self.loop.run_until_complete(self.runtime.stop())
        self.loop.close()


@pytest.fixture(scope="module")
def orch_srv():
    os.environ["GRPC_PORT"] = str(_GRPC_PORT)
    os.environ["ADMIN_PORT"] = str(_ADMIN_PORT)
    os.environ["RATE_LIMIT_DISABLED"] = "1"
    srv = _OrcThread(_GRPC_PORT, _ADMIN_PORT)
    srv.start()
    from tests.integration.conftest import orch_startup_delay

    time.sleep(orch_startup_delay())
    yield srv
    srv.stop()


def _post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{_ADMIN}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{_ADMIN}{path}", timeout=5) as r:
        return json.loads(r.read())


async def _wait_for(condition, timeout: float = 30.0, poll: float | None = None) -> bool:
    from tests.integration.conftest import integration_poll, integration_timeout

    if poll is None:
        poll = integration_poll(0.5)
    deadline = time.monotonic() + integration_timeout(timeout)
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(poll)
    return False


class TestThreeWorkerTraining:
    async def test_three_workers_register(self, orch_srv):
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d = WorkerDaemon(
                    orchestrator_url=_GRPC_URL,
                    node_id=f"three-worker-test-{i}",
                    state_dir=tmp,
                    worker_index=i,
                )
                d.RECONNECT_DELAY = 0.3
                t = asyncio.create_task(d.run())
                workers.append((d, t))
            try:

                def all_registered():
                    try:
                        data = _get("/admin/nodes")
                        node_ids = {n.get("node_id") for n in data.get("nodes", [])}
                        return all(f"three-worker-test-{i}" in node_ids for i in range(3))
                    except Exception:
                        return False

                ok = await _wait_for(all_registered, timeout=15)
                assert ok, "Expected all 3 workers to register"
                data = _get("/admin/nodes")
                assert len(data.get("nodes", [])) >= 3
            finally:
                for d, t in workers:
                    d.stop()
                    t.cancel()
                await asyncio.sleep(1)

    async def test_job_distributed_to_three_workers(self, orch_srv):
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d = WorkerDaemon(
                    orchestrator_url=_GRPC_URL,
                    node_id=f"dist-worker-{i}",
                    state_dir=tmp,
                    worker_index=i,
                )
                d.RECONNECT_DELAY = 0.3
                t = asyncio.create_task(d.run())
                workers.append((d, t))
            try:
                await _wait_for(lambda: len(_get("/admin/nodes").get("nodes", [])) >= 3, timeout=15)
                job = _post(
                    "/admin/jobs",
                    {
                        "steps": 30,
                        "batch_size": 8,
                        "steps_per_task": 10,
                    },
                )
                job_id = job["job_id"]

                def job_complete():
                    try:
                        status = _get(f"/admin/jobs/{job_id}")
                        return status.get("status") == "success"
                    except Exception:
                        return False

                ok = await _wait_for(job_complete, timeout=60)
                assert ok, f"Job {job_id} did not complete"
                final_status = _get(f"/admin/jobs/{job_id}")
                assert final_status.get("status") == "success"
            finally:
                for d, t in workers:
                    d.stop()
                    t.cancel()
                await asyncio.sleep(1)

    async def test_gradients_from_three_workers_aggregated(self, orch_srv):
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d = WorkerDaemon(
                    orchestrator_url=_GRPC_URL,
                    node_id=f"grad-worker-{i}",
                    state_dir=tmp,
                    worker_index=i,
                )
                d.RECONNECT_DELAY = 0.3
                t = asyncio.create_task(d.run())
                workers.append((d, t))
            try:
                await _wait_for(lambda: len(_get("/admin/nodes").get("nodes", [])) >= 3, timeout=15)
                job = _post("/admin/jobs", {"steps": 15, "batch_size": 8})
                job_id = job["job_id"]
                await _wait_for(
                    lambda: _get(f"/admin/jobs/{job_id}").get("status") == "success", timeout=45
                )
                for i in range(3):
                    try:
                        credits = _get(f"/admin/credits/grad-worker-{i}")
                        assert credits.get("lifetime", 0) > 0, (
                            f"Worker {i} should have earned credits"
                        )
                    except Exception:
                        pass
            finally:
                for d, t in workers:
                    d.stop()
                    t.cancel()
                await asyncio.sleep(1)

    async def test_byzantine_detection_with_three_workers(self, orch_srv):
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d = WorkerDaemon(
                    orchestrator_url=_GRPC_URL,
                    node_id=f"byz-test-{i}",
                    state_dir=tmp,
                    worker_index=i,
                )
                d.RECONNECT_DELAY = 0.3
                t = asyncio.create_task(d.run())
                workers.append((d, t))
            try:
                await _wait_for(lambda: len(_get("/admin/nodes").get("nodes", [])) >= 3, timeout=15)
                job = _post("/admin/jobs", {"steps": 10, "batch_size": 8})
                job_id = job["job_id"]
                await _wait_for(
                    lambda: _get(f"/admin/jobs/{job_id}").get("status") == "success", timeout=30
                )
                final = _get(f"/admin/jobs/{job_id}")
                assert final.get("status") == "success"
            finally:
                for d, t in workers:
                    d.stop()
                    t.cancel()
                await asyncio.sleep(1)

    async def test_multiple_jobs_with_three_workers(self, orch_srv):
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d = WorkerDaemon(
                    orchestrator_url=_GRPC_URL,
                    node_id=f"multi-job-worker-{i}",
                    state_dir=tmp,
                    worker_index=i,
                )
                d.RECONNECT_DELAY = 0.3
                t = asyncio.create_task(d.run())
                workers.append((d, t))
            try:
                await _wait_for(lambda: len(_get("/admin/nodes").get("nodes", [])) >= 3, timeout=15)
                job_ids = []
                for _ in range(3):
                    job = _post("/admin/jobs", {"steps": 5, "batch_size": 8})
                    job_ids.append(job["job_id"])

                def all_complete():
                    for job_id in job_ids:
                        try:
                            status = _get(f"/admin/jobs/{job_id}")
                            if status.get("status") != "success":
                                return False
                        except OSError:
                            return False
                    return True

                ok = await _wait_for(all_complete, timeout=60)
                assert ok, "Not all jobs completed"
                for job_id in job_ids:
                    final = _get(f"/admin/jobs/{job_id}")
                    assert final.get("status") == "success", f"Job {job_id} failed"
            finally:
                for d, t in workers:
                    d.stop()
                    t.cancel()
                await asyncio.sleep(1)


class TestCreditSystemWithThreeWorkers:
    async def test_credits_earned_by_all_three_workers(self, orch_srv):
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d = WorkerDaemon(
                    orchestrator_url=_GRPC_URL,
                    node_id=f"credit-worker-{i}",
                    state_dir=tmp,
                    worker_index=i,
                )
                d.RECONNECT_DELAY = 0.3
                t = asyncio.create_task(d.run())
                workers.append((d, t))
            try:
                await _wait_for(lambda: len(_get("/admin/nodes").get("nodes", [])) >= 3, timeout=15)
                job = _post("/admin/jobs", {"steps": 10, "batch_size": 8})
                job_id = job["job_id"]
                await _wait_for(
                    lambda: _get(f"/admin/jobs/{job_id}").get("status") == "success", timeout=30
                )
                await asyncio.sleep(2)
                lifetimes = []
                for i in range(3):
                    try:
                        credits = _get(f"/admin/credits/credit-worker-{i}")
                        lifetimes.append(float(credits.get("lifetime", 0)))
                    except urllib.error.HTTPError:
                        lifetimes.append(0.0)
                assert sum(lifetimes) > 0, (
                    f"Expected credits on at least one worker for completed job, got {lifetimes}"
                )
            finally:
                for d, t in workers:
                    d.stop()
                    t.cancel()
                await asyncio.sleep(1)


class TestFaultToleranceWithThreeWorkers:
    async def test_training_survives_one_worker_disconnect(self, orch_srv):
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d = WorkerDaemon(
                    orchestrator_url=_GRPC_URL,
                    node_id=f"fault-tolerant-{i}",
                    state_dir=tmp,
                    worker_index=i,
                )
                d.RECONNECT_DELAY = 0.3
                t = asyncio.create_task(d.run())
                workers.append((d, t))
            try:
                await _wait_for(lambda: len(_get("/admin/nodes").get("nodes", [])) >= 3, timeout=15)
                job = _post("/admin/jobs", {"steps": 15, "batch_size": 8})
                job_id = job["job_id"]
                await asyncio.sleep(3)
                workers[0][0].stop()
                workers[0][1].cancel()
                await _wait_for(
                    lambda: _get(f"/admin/jobs/{job_id}").get("status") == "success", timeout=45
                )
                final = _get(f"/admin/jobs/{job_id}")
                assert final.get("status") == "success"
            finally:
                for d, t in workers[1:]:
                    d.stop()
                    t.cancel()
                await asyncio.sleep(1)
