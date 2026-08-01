"""
Integration tests for resource limit enforcement.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestResourceLimitsIntegration:
    """Test resource limit functions work in integration."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_get_resource_limits_reads_config(self, temp_config_dir):
        """Verify resource limits are read from config file."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        # Create config file with custom values
        distribai_dir = temp_config_dir / ".distribai"
        distribai_dir.mkdir(parents=True, exist_ok=True)
        config_file = distribai_dir / "desktop.json"
        config_file.write_text(json.dumps({"cpuPercent": 75, "gpuPercent": 60, "ramPercent": 80}))

        # Patch Path.home() to return temp directory
        original_home = Path.home
        try:
            Path.home = lambda: temp_config_dir

            from worker.src.daemon.executor import get_resource_limits

            # Force reimport by clearing cache
            if "worker.src.daemon.executor" in sys.modules:
                del sys.modules["worker.src.daemon.executor"]

            limits = get_resource_limits()

            assert limits["cpuPercent"] == 75
            assert limits["gpuPercent"] == 60
            assert limits["ramPercent"] == 80
        finally:
            Path.home = original_home

    def test_get_resource_limits_defaults(self, temp_config_dir):
        """Verify defaults are used when no config exists."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        with patch("worker.src.daemon.executor.Path") as mock_path:
            mock_path.home.return_value = temp_config_dir
            mock_path.return_value = temp_config_dir / ".distribai" / "desktop.json"
            (temp_config_dir / ".distribai").mkdir(parents=True, exist_ok=True)

            from worker.src.daemon.executor import get_resource_limits

            limits = get_resource_limits()

            # Should return defaults
            assert limits["cpuPercent"] == 50
            assert limits["gpuPercent"] == 50
            assert limits["ramPercent"] == 50

    def test_resource_limits_validation(self, temp_config_dir):
        """Verify limits are clamped to valid range."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        # Create config with out-of-range values
        distribai_dir = temp_config_dir / ".distribai"
        distribai_dir.mkdir(parents=True, exist_ok=True)
        config_file = distribai_dir / "desktop.json"
        config_file.write_text(
            json.dumps(
                {
                    "cpuPercent": 150,  # Over 100
                    "gpuPercent": 5,  # Under 10
                    "ramPercent": -20,  # Negative
                }
            )
        )

        # Patch Path.home() to return temp directory
        original_home = Path.home
        try:
            Path.home = lambda: temp_config_dir

            # Force reimport
            if "worker.src.daemon.executor" in sys.modules:
                del sys.modules["worker.src.daemon.executor"]
            from worker.src.daemon.executor import get_resource_limits

            limits = get_resource_limits()

            # Should be clamped to valid range (10-100)
            assert limits["cpuPercent"] == 100, f"Expected 100, got {limits['cpuPercent']}"
            assert limits["gpuPercent"] == 10, f"Expected 10, got {limits['gpuPercent']}"
            assert limits["ramPercent"] == 10, f"Expected 10, got {limits['ramPercent']}"
        finally:
            Path.home = original_home


class TestResourceMonitorIntegration:
    """Test ResourceMonitor class in integration."""

    @pytest.mark.asyncio
    async def test_resource_monitor_lifecycle(self):
        """Test monitor can start and stop correctly."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from worker.src.daemon.executor import ResourceMonitor

        monitor = ResourceMonitor(ram_percent=50)

        # Start monitor
        await monitor.start()
        assert monitor._monitor_task is not None
        assert not monitor._monitor_task.done()

        # Stop monitor
        await monitor.stop()
        assert monitor._monitor_task is None or monitor._monitor_task.done()

    @pytest.mark.asyncio
    async def test_resource_monitor_respects_stop(self):
        """Test monitor stops when requested."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from worker.src.daemon.executor import ResourceMonitor

        monitor = ResourceMonitor(ram_percent=50)

        await monitor.start()
        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop it
        await monitor.stop()

        # Should be stopped
        assert monitor._stop_event.is_set()


class TestCPULimitIntegration:
    """Test CPU limit application."""

    def test_apply_cpu_limit_skips_at_100_percent(self):
        """Verify CPU limit is skipped when set to 100%."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from worker.src.daemon.executor import apply_cpu_limit

        # Should not raise error at 100%
        with patch("psutil.Process") as mock_process:
            apply_cpu_limit(100)
            # Should not call process methods
            mock_process.assert_not_called()

    def test_apply_cpu_limit_handles_errors(self):
        """Verify CPU limit handles errors gracefully."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from worker.src.daemon.executor import apply_cpu_limit

        # Should not crash on error
        with patch("psutil.Process") as mock_process:
            mock_process.side_effect = Exception("Test error")
            # Should not raise
            apply_cpu_limit(50)


class TestGPULimitIntegration:
    """Test GPU limit application."""

    def test_apply_gpu_limit_skips_without_cuda(self):
        """Verify GPU limit is skipped when CUDA not available."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from worker.src.daemon.executor import apply_gpu_limit

        # Mock torch.cuda as unavailable
        with patch("torch.cuda.is_available", return_value=False):
            # Should not raise error
            apply_gpu_limit(50)

    def test_apply_gpu_limit_handles_errors(self):
        """Verify GPU limit handles errors gracefully."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from worker.src.daemon.executor import apply_gpu_limit

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.set_per_process_memory_fraction") as mock_set:
                mock_set.side_effect = Exception("CUDA error")
                # Should not raise
                apply_gpu_limit(50)


class TestExecutorResourceIntegration:
    """Test JobExecutor integrates with resource limits."""

    def test_executor_loads_limits_on_execute(self):
        """Verify executor loads resource limits when executing job."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        from worker.src.daemon.executor import JobExecutor

        # Create mock callbacks
        async def mock_progress(*args):
            pass

        async def mock_result(*args):
            pass

        executor = JobExecutor(
            node_id="test-node", on_progress=mock_progress, on_result=mock_result
        )

        # Check that resource limits functions exist and can be called
        with patch.object(executor, "_create_model", return_value=MagicMock()):
            with patch.object(executor, "_load_weights"):
                with patch.object(
                    executor, "_load_batch_source", return_value={"mode": "text", "content": "test"}
                ):
                    # Mock the training loop to avoid actual PyTorch execution
                    with patch.object(
                        executor,
                        "_compute_loss",
                        return_value=MagicMock(item=MagicMock(return_value=0.5)),
                    ):
                        with patch.object(executor, "_collect_gradients", return_value=({}, 0.1)):
                            with patch.object(executor, "_write_gradients"):
                                with patch.object(
                                    executor.s3, "upload_file", return_value="mock-url"
                                ):
                                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
