"""
Integration tests for Byzantine fault detection (production API).

Uses the same RobustAggregator / detector classes as the worker; tensors are torch.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "worker" / "src" / "distribai_proto"))
from services_python.orchestrator_grpc import serve as orchestrator_serve
from worker.src.daemon.byzantine_detector import (
    AggregationMethod,
    AnomalyScore,
    Krum,
    RobustAggregator,
    TrimmedMean,
)
from worker.src.daemon.daemon import WorkerDaemon

_GRPC_PORT = 39765
_ADMIN_PORT = 39766
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


class TestByzantineDetectionUnit:
    def test_trimmed_mean_aggregate_shape(self):
        agg = RobustAggregator(method=AggregationMethod.TRIMMED_MEAN)
        updates = {f"n{i}": torch.ones(8) * float(i) for i in range(5)}
        out = agg.aggregate(updates)
        assert out.shape == (8,)

    def test_krum_rejects_extreme_outlier(self):
        agg = RobustAggregator(method=AggregationMethod.KRUM)
        updates = {
            "a": torch.tensor([1.0, 1.0, 1.0]),
            "b": torch.tensor([1.1, 1.1, 1.1]),
            "c": torch.tensor([0.9, 0.9, 0.9]),
            "z": torch.tensor([100.0, 100.0, 100.0]),
        }
        out = agg.aggregate(updates)
        assert out is not None
        assert out.norm() < 10

    def test_trimmed_mean_detect_anomalies_returns_scores(self):
        det = TrimmedMean(max_byzantine_fraction=0.25)
        updates = {f"n{i}": torch.ones(6) for i in range(4)}
        scores = det.detect_anomalies(updates)
        assert len(scores) == 4
        assert all(isinstance(s, AnomalyScore) for s in scores)

    def test_krum_detect_anomalies(self):
        det = Krum(max_byzantine_fraction=0.2)
        scores = det.detect_anomalies(
            {"a": torch.ones(4), "b": torch.ones(4) * 1.05, "c": torch.ones(4) * 0.95}
        )
        assert len(scores) == 3


class TestByzantineIntegration:
    async def test_training_with_byzantine_detection(self, orch_srv):
        with tempfile.TemporaryDirectory() as tmp:
            workers = []
            for i in range(3):
                d = WorkerDaemon(_GRPC_URL, f"byz-test-{i}", tmp, worker_index=i)
                d.RECONNECT_DELAY = 0.3
                t = asyncio.create_task(d.run())
                workers.append((d, t))
            try:
                from tests.integration.conftest import integration_poll, integration_timeout

                await asyncio.sleep(integration_poll(3.0))
                create_req = urllib.request.Request(
                    f"{_ADMIN}/admin/jobs",
                    data=json.dumps({"steps": 10, "batch_size": 8}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(create_req, timeout=5) as r:
                    job = json.loads(r.read())
                    job_id = job["job_id"]
                deadline = time.monotonic() + integration_timeout(30.0)
                poll = integration_poll(0.5)
                while time.monotonic() < deadline:
                    status_req = urllib.request.Request(f"{_ADMIN}/admin/jobs/{job_id}")
                    with urllib.request.urlopen(status_req, timeout=5) as r:
                        status = json.loads(r.read())
                        if status.get("status") == "success":
                            break
                    await asyncio.sleep(poll)
                else:
                    pytest.fail("Job did not complete in time")
                status_req = urllib.request.Request(f"{_ADMIN}/admin/jobs/{job_id}")
                with urllib.request.urlopen(status_req, timeout=5) as r:
                    final_status = json.loads(r.read())
                    assert final_status.get("status") == "success"
            finally:
                for d, t in workers:
                    d.stop()
                    t.cancel()
                await asyncio.sleep(1)
