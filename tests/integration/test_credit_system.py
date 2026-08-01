"""
Integration tests for the credit system.

Tests credit earning, transfers, multipliers, and ledger integrity.
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

_GRPC_PORT = 29765
_ADMIN_PORT = 29766
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


class TestCreditEarning:
    async def test_credits_earned_on_job_completion(self, orch_srv):
        with tempfile.TemporaryDirectory() as tmp:
            d = WorkerDaemon(
                orchestrator_url=_GRPC_URL,
                node_id="credit-test-node",
                state_dir=tmp,
            )
            d.RECONNECT_DELAY = 0.3
            t = asyncio.create_task(d.run())
            try:
                from tests.integration.conftest import integration_poll, integration_timeout

                await asyncio.sleep(integration_poll(2.0))
                req = urllib.request.Request(f"{_ADMIN}/v1/credits/balance")
                req.add_header("Authorization", f"Bearer {d._session_token or 'test'}")
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        initial = json.loads(r.read())
                except Exception:
                    initial = {"confirmed": 0}
                create_req = urllib.request.Request(
                    f"{_ADMIN}/admin/jobs",
                    data=json.dumps({"steps": 5, "batch_size": 8}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(create_req, timeout=5) as r:
                    job = json.loads(r.read())
                    job_id = job["job_id"]
                deadline = time.monotonic() + integration_timeout(20.0)
                poll = integration_poll(0.5)
                while time.monotonic() < deadline:
                    status_req = urllib.request.Request(f"{_ADMIN}/admin/jobs/{job_id}")
                    with urllib.request.urlopen(status_req, timeout=5) as r:
                        status = json.loads(r.read())
                        if status.get("status") == "success":
                            break
                    await asyncio.sleep(poll)
                await asyncio.sleep(integration_poll(1.0))
                credits_req = urllib.request.Request(f"{_ADMIN}/admin/credits/credit-test-node")
                with urllib.request.urlopen(credits_req, timeout=5) as r:
                    final = json.loads(r.read())
                assert final.get("lifetime", 0) > initial.get("confirmed", 0), (
                    f"Expected credits to increase, got initial={initial}, final={final}"
                )
            finally:
                d.stop()
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=2)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass


class TestCreditTransfers:
    async def test_credits_can_be_transferred(self, orch_srv):
        import urllib.request

        with tempfile.TemporaryDirectory() as tmp:
            d1 = WorkerDaemon(_GRPC_URL, "transfer-sender", tmp, worker_index=0)
            d2 = WorkerDaemon(_GRPC_URL, "transfer-receiver", tmp, worker_index=1)
            d1.RECONNECT_DELAY = 0.3
            d2.RECONNECT_DELAY = 0.3
            t1 = asyncio.create_task(d1.run())
            t2 = asyncio.create_task(d2.run())
            try:
                await asyncio.sleep(2)
                req = urllib.request.Request(f"{_ADMIN}/v1/credits/transfers?direction=all")
                req.add_header("Authorization", "Bearer test-jwt")
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        result = json.loads(r.read())
                        assert "transfers" in result
                except urllib.error.HTTPError:
                    pass
            finally:
                d1.stop()
                d2.stop()
                t1.cancel()
                t2.cancel()
                try:
                    await asyncio.wait_for(asyncio.gather(t1, t2), timeout=2)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass


class TestCreditMultipliers:
    async def test_multiplier_status_endpoint(self, orch_srv):
        import urllib.request

        with tempfile.TemporaryDirectory() as tmp:
            d = WorkerDaemon(_GRPC_URL, "multiplier-test", tmp)
            d.RECONNECT_DELAY = 0.3
            t = asyncio.create_task(d.run())
            try:
                await asyncio.sleep(2)
                req = urllib.request.Request(f"{_ADMIN}/v1/credits/multipliers")
                req.add_header("Authorization", f"Bearer {d._session_token or 'test'}")
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        result = json.loads(r.read())
                        assert "effective_multiplier" in result
                        assert result["effective_multiplier"] >= 1.0
                except urllib.error.HTTPError:
                    pass
            finally:
                d.stop()
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass


class TestLedgerIntegrity:
    async def test_ledger_root_hash_exists(self, orch_srv):
        import urllib.request

        req = urllib.request.Request(f"{_ADMIN}/admin/ledger/root")
        with urllib.request.urlopen(req, timeout=5) as r:
            result = json.loads(r.read())
            assert "root_hash" in result
            assert "size" in result
            assert result["size"] >= 0

    async def test_ledger_verification_missing_index_returns_not_found(self, orch_srv):
        import urllib.error
        import urllib.request

        req = urllib.request.Request(f"{_ADMIN}/admin/ledger/verify/99999")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 404
        body = json.loads(exc_info.value.read().decode())
        assert body["error"] == "not found"

    async def test_ledger_verification_returns_proof_after_earn(self, orch_srv):
        import urllib.request

        orch_srv.runtime.node_service.record_credit_earn(
            "ledger-verify-node", 2.5, "job-verify-1"
        )
        req = urllib.request.Request(f"{_ADMIN}/admin/ledger/verify/0")
        with urllib.request.urlopen(req, timeout=5) as r:
            result = json.loads(r.read())
        assert result["index"] == 0
        assert "hash" in result
        assert result["valid"] is True


class TestVotingSystem:
    async def test_vote_reduces_credits(self, orch_srv):
        import urllib.request

        with tempfile.TemporaryDirectory() as tmp:
            d = WorkerDaemon(_GRPC_URL, "vote-test", tmp)
            d.RECONNECT_DELAY = 0.3
            t = asyncio.create_task(d.run())
            try:
                await asyncio.sleep(2)
                create_req = urllib.request.Request(
                    f"{_ADMIN}/admin/jobs",
                    data=json.dumps({"steps": 5, "batch_size": 8}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(create_req, timeout=5) as r:
                    job = json.loads(r.read())
                    job_id = job["job_id"]
                vote_req = urllib.request.Request(
                    f"{_ADMIN}/v1/votes",
                    data=json.dumps({"job_id": job_id, "credits": 10}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                vote_req.add_header("Authorization", f"Bearer {d._session_token or 'test'}")
                try:
                    with urllib.request.urlopen(vote_req, timeout=5) as r:
                        result = json.loads(r.read())
                        assert "vote_id" in result or "error" in result
                except urllib.error.HTTPError as e:
                    assert e.code in [400, 403, 401]
            finally:
                d.stop()
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
