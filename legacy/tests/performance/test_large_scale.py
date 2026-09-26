"""
Large-Scale Performance Tests for DistribAI

Tests system behavior at production scale (100+ nodes).
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

# TEAM_001: Setting up test environment for large-scale testing
os.environ["GRPC_PORT"] = "58765"
os.environ["ADMIN_PORT"] = "58766"
os.environ["RATE_LIMIT_DISABLED"] = "1"
os.environ["JWT_SECRET"] = "test_secret_large_scale_32_bytes_"

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
    """Provide running orchestrator for tests."""
    grpc_port = 58765
    admin_port = 58766

    srv = _OrcThread(grpc_port, admin_port)
    srv.start()
    time.sleep(2)  # Let orchestrator initialize

    yield srv

    srv.stop()


class TestLargeScaleRegistration:
    """Test registration of 100+ simulated nodes."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_100_nodes_register(self, orchestrator):
        """Verify 100 nodes can register successfully."""
        num_nodes = 100
        workers = []
        tmp_dirs = []

        try:
            # Create workers in batches to avoid overwhelming the orchestrator
            batch_size = 20
            for batch in range(0, num_nodes, batch_size):
                batch_workers = []
                for i in range(batch, min(batch + batch_size, num_nodes)):
                    tmpdir = tempfile.mkdtemp(prefix=f"scale-test-{i:03d}-")
                    tmp_dirs.append(tmpdir)

                    daemon = WorkerDaemon(
                        orchestrator_url="127.0.0.1:58765",
                        node_id=f"scale-node-{i:03d}",
                        state_dir=tmpdir,
                    )
                    daemon.RECONNECT_DELAY = 0.1
                    task = asyncio.create_task(daemon.run())
                    batch_workers.append((daemon, task))

                workers.extend(batch_workers)
                await asyncio.sleep(1)  # Let batch connect

            # Wait for all to connect
            timeout = time.time() + 60
            connected = 0

            while time.time() < timeout and connected < num_nodes:
                connected = sum(1 for d, _ in workers if d.connected)
                await asyncio.sleep(0.5)

            assert connected >= num_nodes * 0.95, f"Only {connected}/{num_nodes} nodes connected"

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
    @pytest.mark.timeout(180)
    async def test_100_nodes_heartbeat_sustainability(self, orchestrator):
        """Verify 100 nodes can maintain heartbeats over time."""
        num_nodes = 100
        workers = []
        tmp_dirs = []

        try:
            # Start workers
            for i in range(num_nodes):
                tmpdir = tempfile.mkdtemp(prefix=f"hb-test-{i:03d}-")
                tmp_dirs.append(tmpdir)

                daemon = WorkerDaemon(
                    orchestrator_url="127.0.0.1:58765",
                    node_id=f"hb-node-{i:03d}",
                    state_dir=tmpdir,
                )
                daemon.HEARTBEAT_INTERVAL = 5  # Faster for testing
                task = asyncio.create_task(daemon.run())
                workers.append((daemon, task))

            # Wait for initial connection
            await asyncio.sleep(10)

            # Monitor for 30 seconds
            start_seqs = [d._seq for d, _ in workers]
            await asyncio.sleep(30)
            end_seqs = [d._seq for d, _ in workers]

            # Verify heartbeats progressing
            progressed = sum(1 for s, e in zip(start_seqs, end_seqs, strict=True) if e > s)
            assert progressed >= num_nodes * 0.90, (
                f"Only {progressed}/{num_nodes} nodes showing heartbeat progression"
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


class TestHighThroughputAggregation:
    """Test gradient aggregation at scale."""

    def test_aggregation_performance_100_gradients(self):
        """Test aggregation with 100 simultaneous gradients."""
        import torch

        from worker.src.daemon.byzantine_detector import MultiKrum, TrimmedMean

        num_gradients = 100
        gradient_size = (10000,)  # 10k parameters

        # Generate random gradients
        updates = {f"node-{i:03d}": torch.randn(gradient_size) for i in range(num_gradients)}

        # Test MultiKrum performance
        krum = MultiKrum(max_byzantine_fraction=0.2, device="cpu")

        start = time.time()
        result = krum.aggregate(updates)
        elapsed = time.time() - start

        assert result.shape == gradient_size
        assert elapsed < 10.0, f"MultiKrum aggregation took {elapsed:.2f}s (too slow)"

        # Test TrimmedMean performance
        trimmed = TrimmedMean(max_byzantine_fraction=0.2, device="cpu")

        start = time.time()
        result = trimmed.aggregate(updates)
        elapsed = time.time() - start

        assert result.shape == gradient_size
        assert elapsed < 5.0, f"TrimmedMean aggregation took {elapsed:.2f}s (too slow)"

    def test_batched_aggregation_1000_nodes(self):
        """Test batched aggregation for 1000 nodes."""
        import torch

        from worker.src.daemon.byzantine_detector import AdaptiveAggregator

        num_total = 1000
        batch_size = 50
        gradient_size = (5000,)

        # Generate all gradients
        all_updates = {f"node-{i:04d}": torch.randn(gradient_size) for i in range(num_total)}

        aggregator = AdaptiveAggregator(max_byzantine_fraction=0.2, device="cpu")

        start = time.time()

        # Process in batches
        results = []
        for i in range(0, num_total, batch_size):
            batch = dict(list(all_updates.items())[i : i + batch_size])
            result = aggregator.aggregate(batch)
            results.append(result)

        # Aggregate batch results
        final = torch.stack(results).mean(dim=0)
        elapsed = time.time() - start

        assert final.shape == gradient_size
        assert elapsed < 30.0, f"Batched aggregation took {elapsed:.2f}s (too slow)"


class TestDatabasePerformance:
    """Test database performance under load."""

    def test_concurrent_reads_100_connections(self):
        """Test 100 concurrent read operations."""
        from services_python.db_manager import DBManager

        db_path = ":memory:"
        schema_path = _root / "runtime" / "db" / "schema.sql"
        db = DBManager(str(db_path), str(schema_path))

        # Pre-populate with test data
        for i in range(1000):
            db.create_node(
                node_id=f"perf-node-{i:04d}",
                jwt_token="test",
                hardware_json="{}",
            )

        def read_operation(idx: int) -> dict:
            return db.get_node(f"perf-node-{idx:04d}")

        start = time.time()

        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(read_operation, i % 1000) for i in range(1000)]
            results = [f.result() for f in futures]

        elapsed = time.time() - start

        assert all(r is not None for r in results)
        assert elapsed < 10.0, f"1000 concurrent reads took {elapsed:.2f}s (too slow)"

    def test_write_throughput_1000_operations(self):
        """Test write throughput for 1000 operations."""
        from services_python.db_manager import DBManager

        db_path = ":memory:"
        schema_path = _root / "runtime" / "db" / "schema.sql"
        db = DBManager(str(db_path), str(schema_path))

        start = time.time()

        for i in range(1000):
            db.create_node(
                node_id=f"write-node-{i:04d}",
                jwt_token=f"token-{i}",
                hardware_json=json.dumps({"idx": i}),
            )

        elapsed = time.time() - start

        assert elapsed < 30.0, f"1000 writes took {elapsed:.2f}s (too slow)"


class TestCreditSystemScale:
    """Test credit system at scale."""

    @pytest.mark.asyncio
    async def test_1000_concurrent_transfers(self):
        """Test 1000 concurrent credit transfers."""
        from services_python.credit_transfers import CreditTransferManager
        from services_python.db_manager import DBManager

        db_path = ":memory:"
        schema_path = _root / "runtime" / "db" / "schema.sql"
        db = DBManager(str(db_path), str(schema_path))
        manager = CreditTransferManager(db)

        # Create accounts
        for i in range(100):
            db.update_credit_balance(f"account-{i:03d}", 1000.0)

        start = time.time()

        # Perform 1000 random transfers
        async def random_transfer(idx: int):
            sender = f"account-{random.randint(0, 99):03d}"
            receiver = f"account-{random.randint(0, 99):03d}"
            amount = random.uniform(1, 10)
            try:
                return await manager.transfer(sender, receiver, amount)
            except Exception:
                return None

        tasks = [random_transfer(i) for i in range(1000)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start

        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))

        # At least 80% should succeed (some will fail due to insufficient balance)
        assert successful > 800, f"Only {successful}/1000 transfers succeeded"
        assert elapsed < 60.0, f"1000 transfers took {elapsed:.2f}s (too slow)"


class TestMemoryUsage:
    """Test memory usage at scale."""

    def test_orchestrator_memory_1000_nodes(self):
        """Test orchestrator memory with 1000 registered nodes."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        from services_python.db_manager import DBManager

        db_path = ":memory:"
        schema_path = _root / "runtime" / "db" / "schema.sql"
        db = DBManager(str(db_path), str(schema_path))

        # Register 1000 nodes
        for i in range(1000):
            db.create_node(
                node_id=f"mem-test-{i:04d}",
                jwt_token="test",
                hardware_json=json.dumps({"idx": i, "data": "x" * 1000}),
            )

        # Create service (simulated)
        db.get_all_nodes()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should use less than 500MB for 1000 nodes
        assert memory_increase < 500, f"Memory increased by {memory_increase:.1f}MB (too high)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
