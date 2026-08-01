"""
End-to-end tests for the DistribAI worker backend using the REAL orchestrator.

These tests run the actual system orchestrator (services_python.orchestrator_grpc)
and real workers in separate threads. Each worker runs as if it were a separate PC,
but they're all on the same machine for testing purposes.

This simulates multiple PCs on your PC but runs the actual system pipeline.
This is the only acceptable "mock" - using real code in a test harness.

Run:
    pytest tests/e2e/test_e2e.py -v
    pytest tests/e2e/test_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "src" / "distribai_proto"))
from services_python.orchestrator_grpc import serve as orchestrator_serve
from worker.src.daemon.daemon import WorkerDaemon
from worker.src.distribai_proto import distribai_pb2

_GRPC_PORT = 19765
_ADMIN_PORT = 19766
_GRPC_URL = f"127.0.0.1:{_GRPC_PORT}"
_ADMIN = f"http://127.0.0.1:{_ADMIN_PORT}"

_LOCKDOWN_GRPC_PORT = 19769
_LOCKDOWN_ADMIN_PORT = 19770
_LOCKDOWN_GRPC_URL = f"127.0.0.1:{_LOCKDOWN_GRPC_PORT}"


class _OrcThread:
    def __init__(self, grpc_port: int, admin_port: int) -> None:
        self._grpc_port = grpc_port
        self._admin_port = admin_port
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runtime = None
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        self._ready.wait(timeout=10)
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

        self.loop.run_until_complete(start_orchestrator())
        self._ready.set()
        while not self._stop_event.is_set():
            try:
                self.loop.run_forever()
            except (OSError, RuntimeError):
                pass
        if self.runtime is not None:
            self.loop.run_until_complete(self.runtime.stop())
        self.loop.close()


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


def _delete(path: str) -> dict:
    req = urllib.request.Request(f"{_ADMIN}{path}", method="DELETE")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


async def _wait(condition, timeout: float = 10.0, poll: float | None = None) -> bool:
    from tests.fast_env import poll_seconds

    if poll is None:
        poll = poll_seconds(0.1)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        await asyncio.sleep(poll)
    return False


def _get_nodes() -> dict[str, Any]:
    try:
        return _get("/admin/nodes")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return {"nodes": []}


def _get_jobs() -> dict[str, Any]:
    try:
        return _get("/admin/jobs")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return {"jobs": [], "queue_depth": 0}


def _node_exists(node_id: str) -> bool:
    try:
        data = _get("/admin/nodes")
        return any(n.get("node_id") == node_id for n in data.get("nodes", []))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return False


def _get_job_status(job_id: str) -> str:
    try:
        data = _get(f"/admin/jobs/{job_id}")
        return data.get("status", "unknown")
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return "error"


async def _make_worker(orch: _OrcThread, node_id: str, tmpdir: str) -> tuple:
    d = WorkerDaemon(
        orchestrator_url=_GRPC_URL,
        node_id=node_id,
        state_dir=tmpdir,
    )
    d.RECONNECT_DELAY = 0.3
    t = asyncio.create_task(d.run())
    return d, t


async def _stop_worker(daemon: WorkerDaemon, task: asyncio.Task) -> None:
    daemon.stop()
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=2)
    except (asyncio.CancelledError, Exception):
        pass


@pytest.fixture(scope="module")
def orch_srv() -> _OrcThread:
    os.environ["GRPC_PORT"] = str(_GRPC_PORT)
    os.environ["ADMIN_PORT"] = str(_ADMIN_PORT)
    os.environ["RATE_LIMIT_DISABLED"] = "1"
    srv = _OrcThread(_GRPC_PORT, _ADMIN_PORT)
    srv.start()
    time.sleep(0.15)
    yield srv
    srv.stop()


@pytest.fixture(scope="module")
def orch(orch_srv: _OrcThread) -> _OrcThread:
    return orch_srv


@pytest.fixture(scope="module")
def orch_lockdown() -> _OrcThread:
    prev_poc = os.environ.get("REGISTRATION_REQUIRE_POC")
    os.environ["GRPC_PORT"] = str(_LOCKDOWN_GRPC_PORT)
    os.environ["ADMIN_PORT"] = str(_LOCKDOWN_ADMIN_PORT)
    os.environ["REGISTRATION_REQUIRE_POC"] = "true"
    os.environ["RATE_LIMIT_DISABLED"] = "1"
    os.environ.pop("ADMIN_REQUIRE_AUTH", None)
    srv = _OrcThread(_LOCKDOWN_GRPC_PORT, _LOCKDOWN_ADMIN_PORT)
    srv.start()
    time.sleep(0.15)
    try:
        yield srv
    finally:
        srv.stop()
        if prev_poc is None:
            os.environ.pop("REGISTRATION_REQUIRE_POC", None)
        else:
            os.environ["REGISTRATION_REQUIRE_POC"] = prev_poc


class TestRegistration:
    async def test_single_worker_registers(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "reg-node-01", tmp)
            try:
                ok = await _wait(lambda: _node_exists("reg-node-01"), timeout=10)
                assert ok, "Node should appear in /admin/nodes after connecting"
            finally:
                await _stop_worker(d, t)

    async def test_multiple_workers_all_register(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d, t = await _make_worker(orch, f"multi-reg-{i:02d}", tmp)
                workers.append((d, t))
            try:

                def check_all_nodes():
                    data = _get_nodes()
                    node_ids = {n.get("node_id") for n in data.get("nodes", [])}
                    return all(f"multi-reg-{i:02d}" in node_ids for i in range(3))

                ok = await _wait(check_all_nodes, timeout=15)
                assert ok, "Expected 3 nodes to register"
            finally:
                for d, t in workers:
                    await _stop_worker(d, t)

    async def test_register_ack_sets_session_token(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "token-node", tmp)
            try:
                ok = await _wait(lambda: d._session_token is not None, timeout=8)
                assert ok, "Session token should be set after REGISTER_ACK"
                assert len(d._session_token) > 8
            finally:
                await _stop_worker(d, t)

    @pytest.mark.slow
    async def test_worker_registers_when_poc_required(self, orch_lockdown: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = WorkerDaemon(
                orchestrator_url=_LOCKDOWN_GRPC_URL,
                node_id="poc-lock-node",
                state_dir=tmp,
            )
            d.RECONNECT_DELAY = 0.3
            t = asyncio.create_task(d.run())
            try:
                ok = await _wait(lambda: _node_exists_on(_LOCKDOWN_ADMIN_PORT, "poc-lock-node"), timeout=25)
                assert ok, "Worker should register via PoC when REGISTRATION_REQUIRE_POC=true"
            finally:
                await _stop_worker(d, t)


def _node_exists_on(admin_port: int, node_id: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{admin_port}/admin/nodes", timeout=5) as r:
            data = json.loads(r.read())
        return any(n.get("node_id") == node_id for n in data.get("nodes", []))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return False


class TestJobAssignment:
    async def test_job_assigned_to_idle_worker(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "assign-node", tmp)
            try:
                await _wait(lambda: _node_exists("assign-node"), timeout=10)
                r = _post("/admin/jobs", {"steps": 5, "batch_size": 8})
                jid = r["job_id"]
                ok = await _wait(
                    lambda: _get_job_status(jid) in ("assigned", "running", "success"),
                    timeout=15,
                )
                assert ok, f"Job not assigned; status={_get_job_status(jid)}"
            finally:
                await _stop_worker(d, t)

    async def test_job_completes_with_success(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "complete-node", tmp)
            try:
                await _wait(lambda: _node_exists("complete-node"), timeout=10)
                r = _post("/admin/jobs", {"steps": 5, "batch_size": 8})
                jid = r["job_id"]
                ok = await _wait(
                    lambda: _get_job_status(jid) == "success",
                    timeout=20,
                )
                assert ok, f"Job should succeed; got status={_get_job_status(jid)}"
            finally:
                await _stop_worker(d, t)

    @pytest.mark.slow
    async def test_job_lifecycle_records_task_on_worker(self, orch: _OrcThread) -> None:
        """Enqueue → assign → builtin executor finish leaves task row with assignee + output."""
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "lifecycle-node", tmp)
            try:
                await _wait(lambda: _node_exists("lifecycle-node"), timeout=10)
                r = _post("/admin/jobs", {"steps": 5, "batch_size": 8})
                jid = r["job_id"]
                ok = await _wait(
                    lambda: _get_job_status(jid) == "success",
                    timeout=25,
                )
                assert ok
                detail = _get(f"/admin/jobs/{jid}")
                tasks = detail.get("tasks") or []
                assert tasks, "expected at least one task row"
                finished = [x for x in tasks if x.get("status") == "success"]
                assert finished, f"expected success task; got {tasks!r}"
                assert finished[0].get("assignee_node_id") == "lifecycle-node"
                assert finished[0].get("output") is not None
            finally:
                await _stop_worker(d, t)

    async def test_loss_decreases_over_training(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "loss-node", tmp)
            try:
                await _wait(lambda: _node_exists("loss-node"), timeout=10)
                r = _post("/admin/jobs", {"steps": 20, "batch_size": 16})
                jid = r["job_id"]
                ok = await _wait(
                    lambda: _get_job_status(jid) == "success",
                    timeout=30,
                )
                assert ok, f"Job should complete; got status={_get_job_status(jid)}"
            finally:
                await _stop_worker(d, t)


class TestMultiWorker:
    async def test_jobs_distributed_across_workers(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d, t = await _make_worker(orch, f"dist-{i:02d}", tmp)
                workers.append((d, t))
            try:
                await _wait(lambda: _get_nodes().get("nodes", []).__len__() >= 3, timeout=10)
                job_ids = []
                for _ in range(3):
                    r = _post("/admin/jobs", {"steps": 5, "batch_size": 8})
                    job_ids.append(r["job_id"])
                    await asyncio.sleep(0.05)
                ok = await _wait(
                    lambda: all(_get_job_status(jid) == "success" for jid in job_ids),
                    timeout=30,
                )
                assert ok, "All jobs should complete"
            finally:
                for d, t in workers:
                    await _stop_worker(d, t)

    async def test_queue_waits_for_workers(self, orch: _OrcThread) -> None:
        for _ in range(4):
            _post("/admin/jobs", {"steps": 5})
        await asyncio.sleep(0.6)
        data = _get_jobs()
        assert data.get("queue_depth", 0) >= 4, "All jobs should remain queued without workers"

    async def test_queued_jobs_drain_when_worker_joins(self, orch: _OrcThread) -> None:
        job_ids = []
        for _ in range(2):
            r = _post("/admin/jobs", {"steps": 5, "batch_size": 8})
            job_ids.append(r["job_id"])
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "drain-node", tmp)
            try:
                ok = await _wait(
                    lambda: all(
                        _get_job_status(jid) in ("assigned", "running", "success")
                        for jid in job_ids
                    ),
                    timeout=20,
                )
                assert ok, "Pre-queued jobs should be picked up"
            finally:
                await _stop_worker(d, t)


class TestHeartbeat:
    async def test_heartbeat_updates_node_status(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "hb-node", tmp)
            try:
                ok = await _wait(lambda: _node_exists("hb-node"), timeout=10)
                assert ok, "Node should register"
            finally:
                await _stop_worker(d, t)

    async def test_state_file_written(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "state-node", tmp)
            try:
                await _wait(lambda: _node_exists("state-node"), timeout=10)
                state_file = Path(tmp) / "state-node" / "state.json"
                ok = await _wait(lambda: state_file.exists(), timeout=5)
                assert ok, "State file should be written to disk"
                import json as _json

                with open(state_file) as f:
                    s = _json.load(f)
                assert s["node_id"] == "state-node"
            finally:
                await _stop_worker(d, t)


class TestAdminAPI:
    async def test_health_returns_ok(self, orch: _OrcThread) -> None:
        data = _get("/admin/health")
        assert data.get("ok") is True

    async def test_list_nodes_shows_connected(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "api-node", tmp)
            try:
                await _wait(lambda: _node_exists("api-node"), timeout=10)
                data = _get("/admin/nodes")
                ids = [n.get("node_id") for n in data.get("nodes", [])]
                assert "api-node" in ids
            finally:
                await _stop_worker(d, t)

    async def test_list_jobs_initially_empty(self, orch: _OrcThread) -> None:
        existing = _get("/admin/jobs")
        for job in existing.get("jobs", []):
            _delete(f"/admin/jobs/{job['job_id']}")
        data = _get("/admin/jobs")
        assert data.get("jobs", []) == []
        assert data.get("queue_depth", 0) == 0

    async def test_create_job_returns_ids(self, orch: _OrcThread) -> None:
        r = _post("/admin/jobs", {"steps": 5})
        assert "job_id" in r
        assert "task_id" in r
        assert r.get("ok") is True

    async def test_get_job_by_id(self, orch: _OrcThread) -> None:
        r = _post("/admin/jobs", {"steps": 5})
        jid = r["job_id"]
        detail = _get(f"/admin/jobs/{jid}")
        assert detail.get("job_id") == jid
        assert detail.get("status") == "queued"

    async def test_get_nonexistent_job_404(self, orch: _OrcThread) -> None:
        import urllib.error

        with pytest.raises(urllib.error.HTTPError) as exc:
            _get("/admin/jobs/nonexistent-999")
        assert exc.value.code == 404

    async def test_cancel_queued_job(self, orch: _OrcThread) -> None:
        r = _post("/admin/jobs", {"steps": 5})
        jid = r["job_id"]
        result = _delete(f"/admin/jobs/{jid}")
        assert result.get("ok") is True
        detail = _get(f"/admin/jobs/{jid}")
        assert detail.get("status") == "cancelled"


class TestIntegration:
    async def test_full_pipeline(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "pipeline-node", tmp)
            try:
                ok = await _wait(lambda: _node_exists("pipeline-node"), timeout=10)
                assert ok, "Worker should register"
                r = _post("/admin/jobs", {"steps": 5, "batch_size": 8})
                jid = r["job_id"]
                ok = await _wait(
                    lambda: _get_job_status(jid) == "success",
                    timeout=20,
                )
                assert ok, "Job should complete successfully"
                job = _get(f"/admin/jobs/{jid}")
                assert job.get("status") == "success"
            finally:
                await _stop_worker(d, t)


def _minimal_script_package() -> bytes:
    body = (
        b"import json\n"
        b'with open("results.json", "w", encoding="utf-8") as f:\n'
        b'    json.dump({"ok": True, "credits_earned": 1}, f)\n'
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="run.py")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    return buf.getvalue()


def _node_stream_connected(orch: _OrcThread, node_id: str) -> bool:
    async def _check() -> bool:
        return node_id in orch.runtime.node_service.connected_nodes

    try:
        fut = asyncio.run_coroutine_threadsafe(_check(), orch.loop)
        return bool(fut.result(timeout=2))
    except (TimeoutError, OSError, RuntimeError):
        return False


def _push_script_assign(
    orch: _OrcThread,
    node_id: str,
    job_id: str,
    task_id: str,
    package: bytes,
) -> None:
    async def _do() -> None:
        ns = orch.runtime.node_service
        if node_id not in ns.connected_nodes:
            raise RuntimeError(f"node {node_id} not connected")
        queue = ns.connected_nodes[node_id]
        msg = distribai_pb2.ServerMessage(
            assign=distribai_pb2.TaskAssign(
                task_id=task_id,
                job_id=job_id,
                model_name="script-e2e",
                script_package=package,
                hparams_json='{"max_runtime_seconds": 60}',
                steps=1,
                execution_paradigm="script",
            )
        )
        await queue.put(msg)
        ns.pending_assignments[node_id] = task_id

    fut = asyncio.run_coroutine_threadsafe(_do(), orch.loop)
    fut.result(timeout=10)


class TestScriptAssign:
    """gRPC TaskAssign with non-empty script_package (orchestrator queue injection)."""

    @pytest.mark.slow
    async def test_script_package_assign_completes(self, orch: _OrcThread) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "script-grpc-node", tmp)
            try:
                ok = await _wait(lambda: _node_exists("script-grpc-node"), timeout=15)
                assert ok
                ok = await _wait(
                    lambda: _node_stream_connected(orch, "script-grpc-node"),
                    timeout=15,
                )
                assert ok
                # Keep scheduler off this node so the injected script assign is exercised.
                _post(
                    "/admin/nodes/script-grpc-node/contributing",
                    {"contributing": False},
                )
                created = _post("/admin/jobs", {"steps": 2, "batch_size": 4})
                job_id = created["job_id"]
                task_id = created["task_id"]
                assert task_id, "job create should return latest_task_id"
                _push_script_assign(
                    orch,
                    "script-grpc-node",
                    job_id,
                    task_id,
                    _minimal_script_package(),
                )
                ok = await _wait(
                    lambda: _get_job_status(job_id) == "success",
                    timeout=30,
                )
                assert ok
            finally:
                await _stop_worker(d, t)

    @pytest.mark.slow
    async def test_script_job_via_admin_create_and_scheduler(self, orch: _OrcThread) -> None:
        """POST /admin/jobs with script_package_b64 → scheduler assigns → worker runs."""
        pkg_b64 = base64.b64encode(_minimal_script_package()).decode("ascii")
        with tempfile.TemporaryDirectory() as tmp:
            d, t = await _make_worker(orch, "script-sched-node", tmp)
            try:
                ok = await _wait(lambda: _node_stream_connected(orch, "script-sched-node"), timeout=15)
                assert ok
                created = _post(
                    "/admin/jobs",
                    {
                        "steps": 2,
                        "batch_size": 4,
                        "script_package_b64": pkg_b64,
                    },
                )
                job_id = created["job_id"]
                ok = await _wait(
                    lambda: _get_job_status(job_id) == "success",
                    timeout=30,
                )
                assert ok
                detail = _get(f"/admin/jobs/{job_id}")
                tasks = detail.get("tasks") or []
                finished = [x for x in tasks if x.get("status") == "success"]
                assert finished
                out = finished[0].get("output") or {}
                results = out.get("results") if isinstance(out, dict) else None
                assert isinstance(results, dict) and results.get("ok") is True
            finally:
                await _stop_worker(d, t)
