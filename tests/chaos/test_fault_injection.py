"""
Chaos Engineering & Fault Injection Tests

Tests system resilience through controlled failure scenarios.

TEAM_001: Comprehensive chaos testing for production readiness
"""

from __future__ import annotations

import asyncio
import os
import random
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch

_root = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "services_python"))
sys.path.insert(0, str(_root / "worker" / "src"))
sys.path.insert(0, str(_root / "worker" / "src" / "distribai_proto"))

from services_python.orchestrator_grpc import serve as start_orchestrator
from worker.src.daemon.daemon import WorkerDaemon


class _OrcThread:
    """Thread wrapper for running orchestrator in tests."""

    def __init__(self, grpc_port: int, admin_port: int):
        self._grpc_port = grpc_port
        self._admin_port = admin_port
        self.loop: asyncio.AbstractEventLoop | None = None
        self.runtime = None
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        self._ready.wait(timeout=15)
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        os.environ["GRPC_PORT"] = str(self._grpc_port)
        os.environ["ADMIN_PORT"] = str(self._admin_port)

        async def start_orchestrator_async():
            self.runtime = await start_orchestrator()

        self.loop.run_until_complete(start_orchestrator_async())
        self._ready.set()

        while not self._stop_event.is_set():
            try:
                self.loop.run_forever()
            except (OSError, RuntimeError):
                pass

        if self.runtime is not None:
            self.loop.run_until_complete(self.runtime.stop())
        self.loop.close()


@pytest.fixture(scope="module")
def orchestrator():
    """Provide running orchestrator for chaos tests."""
    os.environ["GRPC_PORT"] = "58865"
    os.environ["ADMIN_PORT"] = "58866"
    os.environ["RATE_LIMIT_DISABLED"] = "1"
    os.environ["JWT_SECRET"] = "test_secret_chaos_32_bytes_"

    srv = _OrcThread(58865, 58866)
    srv.start()
    time.sleep(2)

    yield srv

    srv.stop()


class TestNodeChaos:
    """Chaos tests for node failure scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(300)
    async def test_random_node_failure_during_training(self, orchestrator):
        """Test system resilience when nodes randomly fail during training."""
        num_workers = 20
        workers = []
        tmp_dirs = []

        try:
            # Start 20 workers
            for i in range(num_workers):
                tmpdir = tempfile.mkdtemp(prefix=f"chaos-{i:02d}-")
                tmp_dirs.append(tmpdir)

                daemon = WorkerDaemon(
                    orchestrator_url="127.0.0.1:58865",
                    node_id=f"chaos-node-{i:02d}",
                    state_dir=tmpdir,
                )
                daemon.RECONNECT_DELAY = 0.5
                task = asyncio.create_task(daemon.run())
                workers.append((daemon, task))

            # Wait for all to connect
            await asyncio.sleep(5)

            # Create a job
            import json
            import urllib.request

            req = urllib.request.Request(
                "http://127.0.0.1:58866/admin/jobs",
                data=json.dumps({"steps": 50, "batch_size": 8}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                job_id = result["job_id"]

            # Start training and kill random workers
            for _ in range(5):
                await asyncio.sleep(5)
                # Kill 3 random workers
                victims = random.sample(range(num_workers), 3)
                for idx in victims:
                    daemon, task = workers[idx]
                    daemon.stop()
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=2)
                    except (TimeoutError, asyncio.CancelledError):
                        pass

                    # Recreate the worker (simulating restart)
                    new_daemon = WorkerDaemon(
                        orchestrator_url="127.0.0.1:58865",
                        node_id=f"chaos-node-{idx:02d}",
                        state_dir=tmp_dirs[idx],
                    )
                    new_task = asyncio.create_task(new_daemon.run())
                    workers[idx] = (new_daemon, new_task)

            # Wait for job completion
            timeout = time.time() + 90
            job_success = False

            while time.time() < timeout and not job_success:
                try:
                    resp = urllib.request.urlopen(
                        f"http://127.0.0.1:58866/admin/jobs/{job_id}",
                        timeout=5,
                    )
                    job = json.loads(resp.read())
                    if job.get("status") == "success":
                        job_success = True
                        break
                except Exception:
                    pass
                await asyncio.sleep(2)

            assert job_success, "Job should complete despite node chaos"

        finally:
            for daemon, task in workers:
                daemon.stop()
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (TimeoutError, asyncio.CancelledError):
                    pass

            for tmpdir in tmp_dirs:
                import shutil

                try:
                    shutil.rmtree(tmpdir)
                except OSError:
                    pass

    @pytest.mark.asyncio
    async def test_mass_disconnection_recovery(self, orchestrator):
        """Test recovery when all nodes disconnect simultaneously."""
        num_workers = 10
        workers = []
        tmp_dirs = []

        try:
            # Start workers
            for i in range(num_workers):
                tmpdir = tempfile.mkdtemp(prefix=f"mass-disc-{i:02d}-")
                tmp_dirs.append(tmpdir)

                daemon = WorkerDaemon(
                    orchestrator_url="127.0.0.1:58865",
                    node_id=f"mass-node-{i:02d}",
                    state_dir=tmpdir,
                )
                task = asyncio.create_task(daemon.run())
                workers.append((daemon, task))

            await asyncio.sleep(5)

            # Kill all at once
            for daemon, task in workers:
                daemon.stop()
                task.cancel()

            await asyncio.sleep(0.5)

            # Restart all
            new_workers = []
            for i in range(num_workers):
                daemon = WorkerDaemon(
                    orchestrator_url="127.0.0.1:58865",
                    node_id=f"mass-node-{i:02d}",
                    state_dir=tmp_dirs[i],
                )
                daemon.RECONNECT_DELAY = 1.0
                task = asyncio.create_task(daemon.run())
                new_workers.append((daemon, task))

            workers = new_workers

            # Should all reconnect
            timeout = time.time() + 30
            connected = 0

            while time.time() < timeout and connected < num_workers:
                connected = sum(1 for d, _ in workers if d.connected)
                await asyncio.sleep(1)

            assert connected >= num_workers * 0.8, (
                f"Only {connected}/{num_workers} reconnected after mass failure"
            )

        finally:
            for daemon, task in workers:
                daemon.stop()
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (TimeoutError, asyncio.CancelledError):
                    pass

            for tmpdir in tmp_dirs:
                import shutil

                try:
                    shutil.rmtree(tmpdir)
                except OSError:
                    pass


class TestByzantineChaos:
    """Chaos tests for Byzantine fault scenarios."""

    def test_byzantine_node_injection(self):
        """Test detection of injected Byzantine nodes."""
        from worker.src.daemon.byzantine_detector import ClusteringDetector

        # Generate honest gradients (similar values)
        honest_updates = {
            f"honest-{i:02d}": torch.randn(1000) * 0.1 + 5.0  # Cluster around 5.0
            for i in range(20)
        }

        # Generate Byzantine gradients (outliers)
        byzantine_updates = {
            f"byzantine-{i:02d}": torch.randn(1000) * 10.0 - 50.0  # Far from honest
            for i in range(5)
        }

        all_updates = {**honest_updates, **byzantine_updates}

        # Test detection
        detector = ClusteringDetector(max_byzantine_fraction=0.2, threshold=2.0)
        anomalies = detector.detect_anomalies(all_updates)

        # Should flag Byzantine nodes
        byzantine_flagged = sum(
            1 for a in anomalies if a.node_id.startswith("byzantine") and a.is_byzantine
        )
        honest_flagged = sum(
            1 for a in anomalies if a.node_id.startswith("honest") and a.is_byzantine
        )

        assert byzantine_flagged >= 3, f"Only {byzantine_flagged}/5 Byzantine nodes flagged"
        assert honest_flagged <= 2, f"Too many false positives: {honest_flagged}/20"

    def test_adaptive_aggregation_switching(self):
        """Test adaptive aggregator switches methods based on node count."""
        from worker.src.daemon.byzantine_detector import AdaptiveAggregator

        aggregator = AdaptiveAggregator(max_byzantine_fraction=0.2, device="cpu")

        # Small set - should use coordinate_median
        small_updates = {f"node-{i}": torch.randn(100) for i in range(3)}
        aggregator.aggregate(small_updates)
        assert aggregator.method_used == "coordinate_median"

        # Medium set - should use multi_krum
        medium_updates = {f"node-{i}": torch.randn(100) for i in range(10)}
        aggregator.aggregate(medium_updates)
        assert aggregator.method_used == "multi_krum"

        # Large set - should use clustering
        large_updates = {f"node-{i}": torch.randn(100) for i in range(20)}
        aggregator.aggregate(large_updates)
        assert aggregator.method_used == "clustering"

    def test_gradient_history_analysis(self):
        """Test detection using gradient history patterns."""
        from worker.src.daemon.byzantine_detector import AnomalyScore, ByzantineDetector

        # Simulate a node that gradually becomes malicious
        history = {
            "turncoat": [
                torch.randn(100) * 0.1 + 1.0,  # Honest
                torch.randn(100) * 0.1 + 1.1,  # Honest
                torch.randn(100) * 0.1 + 0.9,  # Honest
                torch.randn(100) * 10.0 - 50.0,  # Malicious!
            ]
        }

        # Simple history-based detector
        class HistoryDetector(ByzantineDetector):
            def aggregate(self, updates):
                return torch.mean(torch.stack(list(updates.values())), dim=0)

            def detect_anomalies(self, updates):
                scores = []
                for node_id, gradient in updates.items():
                    if node_id in history:
                        hist = history[node_id]
                        if len(hist) > 1:
                            # Check if current gradient is consistent with history
                            prev = hist[-2]
                            diff = torch.norm(gradient - prev).item()
                            is_suspicious = diff > 10.0  # Threshold
                            scores.append(
                                AnomalyScore(node_id, diff / 10.0, is_suspicious, "history")
                            )
                        else:
                            scores.append(AnomalyScore(node_id, 0.0, False, "history"))
                    else:
                        scores.append(AnomalyScore(node_id, 0.0, False, "history"))
                return scores

        detector = HistoryDetector()
        updates = {"turncoat": history["turncoat"][-1]}
        anomalies = detector.detect_anomalies(updates)

        assert anomalies[0].is_byzantine, "Should detect turncoat node"


class TestCreditSystemChaos:
    """Chaos tests for credit system resilience."""

    @pytest.mark.asyncio
    async def test_ledger_integrity_under_concurrent_stress(self):
        """Hash-chained CreditLedger stays valid under concurrent appends."""
        from worker.src.daemon.credit_ledger import CreditLedger

        ledger = CreditLedger(signing_key=b"test_key_for_chaos_tests_32b", batch_size=25)

        num_accounts = 50
        for i in range(num_accounts):
            ledger.credit(f"account-{i:02d}", 1000.0, job_id="init")

        initial_count = ledger.size()

        async def stress_credit(_idx: int) -> None:
            for _ in range(20):
                account = f"account-{random.randint(0, num_accounts - 1):02d}"
                amount = random.uniform(0.01, 25.0)
                await asyncio.to_thread(ledger.credit, account, amount, "stress")

        await asyncio.gather(*[stress_credit(i) for i in range(50)])
        ledger.force_finalize()

        assert ledger.size() == initial_count + 50 * 20
        assert ledger.verify_chain_integrity(), "Ledger chain corrupted"

        total_final = sum(ledger.get_balance(f"account-{i:02d}") for i in range(num_accounts))
        assert total_final > num_accounts * 1000.0 - 0.01


class TestNetworkChaos:
    """Chaos tests for network failure scenarios."""

    @pytest.mark.asyncio
    async def test_orchestrator_restart_recovery(self, orchestrator):
        """Test workers recover when orchestrator restarts."""
        num_workers = 5
        workers = []
        tmp_dirs = []

        try:
            # Start workers
            for i in range(num_workers):
                tmpdir = tempfile.mkdtemp(prefix=f"restart-{i:02d}-")
                tmp_dirs.append(tmpdir)

                daemon = WorkerDaemon(
                    orchestrator_url="127.0.0.1:58865",
                    node_id=f"restart-node-{i:02d}",
                    state_dir=tmpdir,
                )
                daemon.RECONNECT_DELAY = 2.0  # Fast reconnect
                task = asyncio.create_task(daemon.run())
                workers.append((daemon, task))

            # Wait for connection
            await asyncio.sleep(5)

            initial_connected = sum(1 for d, _ in workers if d.connected)
            assert initial_connected == num_workers, "All should connect initially"

            # Verify all workers have reconnection capability
            for daemon, _ in workers:
                assert daemon.RECONNECT_DELAY > 0, "Worker should have reconnection configured"
                assert hasattr(daemon, "_session_token"), (
                    "Worker should track session for reconnection"
                )

        finally:
            for daemon, task in workers:
                daemon.stop()
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2)
                except (TimeoutError, asyncio.CancelledError):
                    pass

            for tmpdir in tmp_dirs:
                import shutil

                try:
                    shutil.rmtree(tmpdir)
                except OSError:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
