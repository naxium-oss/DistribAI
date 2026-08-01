"""Tests for memory management utilities."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import the module under test
import services_python.memory_manager as memory_manager_module


class TestMemoryManager:
    """Tests for the MemoryManager class."""

    @pytest.fixture
    def manager(self):
        """Create a MemoryManager instance for testing."""
        return memory_manager_module.MemoryManager(
            memory_threshold_gb=1.0,
            cleanup_interval=1,  # Short interval for testing
        )

    def test_init(self, manager):
        """Test MemoryManager initialization."""
        assert manager.memory_threshold_gb == 1.0
        assert manager.cleanup_interval == 1
        assert manager._cleanup_task is None
        assert manager._memory_history == []

    def test_get_memory_info(self, manager):
        """Test getting memory information."""
        with patch("psutil.virtual_memory") as mock_mem, patch("psutil.Process") as mock_process:
            # Mock system memory
            mock_mem.return_value = Mock()
            mock_mem.return_value.percent = 75.0
            mock_mem.return_value.used = 8 * 1024**3  # 8GB
            mock_mem.return_value.available = 4 * 1024**3  # 4GB
            mock_mem.return_value.total = 12 * 1024**3  # 12GB

            # Mock process memory
            mock_proc = Mock()
            mock_proc.memory_info.return_value = Mock()
            mock_proc.memory_info.return_value.rss = 500 * 1024**2  # 500MB
            mock_proc.memory_info.return_value.vms = 800 * 1024**2  # 800MB
            mock_process.return_value = mock_proc

            info = manager.get_memory_info()

            assert "system_memory" in info
            assert "process_memory" in info
            assert info["system_memory"]["percent"] == 75.0
            assert info["system_memory"]["used_gb"] == 8.0
            assert info["system_memory"]["available_gb"] == 4.0
            assert info["process_memory"]["rss_mb"] == 500.0

    def test_get_memory_info_with_gpu(self, manager):
        """Test getting memory information with GPU."""
        with (
            patch("psutil.virtual_memory") as mock_mem,
            patch("psutil.Process") as mock_process,
            patch("services_python.memory_manager.torch") as mock_torch,
        ):
            # Mock torch availability
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3  # 2GB
            mock_torch.cuda.memory_reserved.return_value = 3 * 1024**3  # 3GB
            mock_torch.cuda.get_device_properties.return_value = Mock()
            mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3  # 8GB

            # Mock system memory
            mock_mem.return_value = Mock()
            mock_mem.return_value.percent = 50.0
            mock_mem.return_value.used = 6 * 1024**3
            mock_mem.return_value.available = 6 * 1024**3
            mock_mem.return_value.total = 12 * 1024**3

            # Mock process memory
            mock_proc = Mock()
            mock_proc.memory_info.return_value = Mock()
            mock_proc.memory_info.return_value.rss = 500 * 1024**2
            mock_proc.memory_info.return_value.vms = 800 * 1024**2
            mock_process.return_value = mock_proc

            info = manager.get_memory_info()

            assert "gpu_memory" in info
            assert info["gpu_memory"]["allocated_gb"] == 2.0
            assert info["gpu_memory"]["cached_gb"] == 3.0
            assert info["gpu_memory"]["total_gb"] == 8.0
            assert info["gpu_memory"]["utilization_percent"] == 25.0

    @pytest.mark.asyncio
    async def test_check_and_cleanup_no_cleanup_needed(self, manager):
        """Test memory check when no cleanup is needed."""
        with (
            patch.object(manager, "get_memory_info") as mock_get_info,
            patch.object(manager, "perform_cleanup") as mock_cleanup,
        ):
            # Mock memory info with plenty of available memory
            mock_get_info.return_value = {
                "system_memory": {"available_gb": 5.0},
                "gpu_memory": {"total_gb": 8.0, "allocated_gb": 2.0},  # 6GB available
            }

            result = await manager.check_and_cleanup()

            assert result is False
            mock_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_cleanup_system_memory_low(self, manager):
        """Test memory check when system memory is low."""
        with (
            patch.object(manager, "get_memory_info") as mock_get_info,
            patch.object(manager, "perform_cleanup") as mock_cleanup,
        ):
            # Mock memory info with low system memory
            mock_get_info.return_value = {
                "system_memory": {"available_gb": 0.5},  # Below threshold
                "gpu_memory": {"total_gb": 8.0, "allocated_gb": 2.0},
            }

            mock_cleanup.return_value = None

            result = await manager.check_and_cleanup()

            assert result is True
            mock_cleanup.assert_called_once_with("system_memory_low")

    @pytest.mark.asyncio
    async def test_check_and_cleanup_gpu_memory_low(self, manager):
        """Test memory check when GPU memory is low."""
        with (
            patch.object(manager, "get_memory_info") as mock_get_info,
            patch.object(manager, "perform_cleanup") as mock_cleanup,
        ):
            # Mock memory info with low GPU memory
            mock_get_info.return_value = {
                "system_memory": {"available_gb": 5.0},
                "gpu_memory": {"total_gb": 4.0, "allocated_gb": 3.5},  # 0.5GB available
            }

            mock_cleanup.return_value = None

            result = await manager.check_and_cleanup()

            assert result is True
            mock_cleanup.assert_called_once_with("gpu_memory_low")

    @pytest.mark.asyncio
    async def test_perform_cleanup(self, manager):
        """Test memory cleanup performance."""
        with (
            patch("gc.collect") as mock_gc,
            patch("services_python.memory_manager.torch") as mock_torch,
        ):
            # Mock torch
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.return_value = 1024**3  # 1GB
            mock_torch.cuda.empty_cache.return_value = None
            mock_torch.cuda.reset_peak_memory_stats.return_value = None
            mock_torch.mps.empty_cache = Mock()

            mock_gc.return_value = 42

            await manager.perform_cleanup("test")

            mock_gc.assert_called()
            mock_torch.cuda.empty_cache.assert_called()
            mock_torch.cuda.reset_peak_memory_stats.assert_called()

    @pytest.mark.asyncio
    async def test_handle_oom_with_retry_success(self, manager):
        """Test OOM handling with successful retry."""
        operation = Mock(return_value="success")

        result = await manager.handle_oom_with_retry(operation, "test_operation")

        assert result == "success"
        operation.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_oom_with_retry_oom_recovery(self, manager):
        """Test OOM handling with recovery after OOM error."""
        operation = Mock()
        operation.side_effect = [
            RuntimeError("out of memory"),  # First call fails with OOM
            "success",  # Second call succeeds
        ]

        with patch.object(manager, "perform_cleanup") as mock_cleanup:
            mock_cleanup.return_value = None

            result = await manager.handle_oom_with_retry(operation, "test_operation")

            assert result == "success"
            assert operation.call_count == 2
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_oom_with_retry_max_retries_exceeded(self, manager):
        """Test OOM handling when max retries are exceeded."""
        operation = Mock()
        operation.side_effect = RuntimeError("out of memory")

        with patch.object(manager, "perform_cleanup") as mock_cleanup:
            mock_cleanup.return_value = None

            with pytest.raises(RuntimeError, match="out of memory"):
                await manager.handle_oom_with_retry(operation, "test_operation", max_retries=1)

            assert operation.call_count == 2  # Original + 1 retry
            assert mock_cleanup.call_count == 1

    def test_is_oom_error_torch_cuda(self, manager):
        """Test OOM error detection for PyTorch CUDA."""
        with patch("services_python.memory_manager.torch") as mock_torch:
            mock_torch.cuda.OutOfMemoryError = Exception
            oom_error = mock_torch.cuda.OutOfMemoryError("CUDA out of memory")

            assert manager._is_oom_error(oom_error) is True

    def test_is_oom_error_runtime_oom(self, manager):
        """Test OOM error detection for RuntimeError with OOM message."""
        oom_error = RuntimeError("CUDA out of memory")
        non_oom_error = RuntimeError("Other error")

        assert manager._is_oom_error(oom_error) is True
        assert manager._is_oom_error(non_oom_error) is False

    def test_is_oom_error_memory_error(self, manager):
        """Test OOM error detection for MemoryError."""
        memory_error = MemoryError("Out of memory")

        assert manager._is_oom_error(memory_error) is True

    def test_get_memory_trend_no_data(self, manager):
        """Test memory trend with no historical data."""
        trend = manager.get_memory_trend()

        assert trend["trend"] == "no_data"
        assert trend["data_points"] == 0

    def test_get_memory_trend_with_data(self, manager):
        """Test memory trend with historical data."""
        # Add some historical data
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        for i in range(10):
            timestamp = now - timedelta(minutes=i)
            manager._memory_history.append(
                {"timestamp": timestamp.isoformat(), "system_memory": {"percent": 50 + i}}
            )

        trend = manager.get_memory_trend(minutes=60)

        assert trend["trend"] in ["increasing", "decreasing", "stable"]
        assert trend["data_points"] == 10
        assert "average_percent" in trend
        assert "max_percent" in trend
        assert "min_percent" in trend

    def test_get_memory_health_score_excellent(self, manager):
        """Test memory health score calculation for excellent memory."""
        with patch.object(manager, "get_memory_info") as mock_get_info:
            mock_get_info.return_value = {
                "system_memory": {"percent": 30.0},
                "gpu_memory": {"utilization_percent": 40.0},
            }

            score = manager.get_memory_health_score()

            assert score == 100.0

    def test_get_memory_health_score_warning(self, manager):
        """Test memory health score calculation for warning level."""
        with patch.object(manager, "get_memory_info") as mock_get_info:
            mock_get_info.return_value = {
                "system_memory": {"percent": 75.0},
                "gpu_memory": {"utilization_percent": 70.0},
            }

            score = manager.get_memory_health_score()

            assert 50 <= score <= 80

    def test_get_memory_health_score_critical(self, manager):
        """Test memory health score calculation for critical memory."""
        with patch.object(manager, "get_memory_info") as mock_get_info:
            mock_get_info.return_value = {
                "system_memory": {"percent": 96.0},
                "gpu_memory": {"utilization_percent": 98.0},
            }

            score = manager.get_memory_health_score()

            assert score < 30

    @pytest.mark.asyncio
    async def test_start_background_cleanup(self, manager):
        """Test starting background cleanup task."""
        mock_task = Mock()

        def _consume_coro_and_return_mock(coro):
            # Avoid RuntimeWarning: coroutine was never awaited (patch replaces real create_task).
            coro.close()
            return mock_task

        with patch(
            "asyncio.create_task", side_effect=_consume_coro_and_return_mock
        ) as mock_create_task:
            await manager.start_background_cleanup()

            assert manager._cleanup_task == mock_task
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_background_cleanup(self, manager):
        """Test stopping background cleanup task."""
        mock_task = asyncio.Future()
        manager._cleanup_task = mock_task

        await manager.stop_background_cleanup()

        assert mock_task.cancelled()


class TestMemoryManagerModule:
    """Tests for memory manager module functions."""

    def test_get_memory_manager(self):
        """Test getting the global memory manager."""
        manager = memory_manager_module.get_memory_manager()
        assert isinstance(manager, memory_manager_module.MemoryManager)

        # Should return the same instance
        manager2 = memory_manager_module.get_memory_manager()
        assert manager is manager2

    @pytest.mark.asyncio
    async def test_handle_oom_with_retry_convenience(self):
        """Test convenience function for OOM handling."""
        operation = Mock(return_value="success")

        with patch(
            "services_python.memory_manager._memory_manager.handle_oom_with_retry",
            new_callable=AsyncMock,
        ) as mock_handle:
            mock_handle.return_value = "success"
            result = await memory_manager_module.handle_oom_with_retry(operation, "test")

            assert result == "success"
            mock_handle.assert_called_once_with(operation, "test", 2)

    def test_get_memory_info_convenience(self):
        """Test convenience function for getting memory info."""
        with patch(
            "services_python.memory_manager._memory_manager.get_memory_info"
        ) as mock_get_info:
            mock_get_info.return_value = {"test": "data"}

            result = memory_manager_module.get_memory_info()

            assert result == {"test": "data"}
            mock_get_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_and_cleanup_memory_convenience(self):
        """Test convenience function for memory check and cleanup."""
        with patch(
            "services_python.memory_manager._memory_manager.check_and_cleanup",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.return_value = True

            result = await memory_manager_module.check_and_cleanup_memory()

            assert result is True
            mock_check.assert_called_once()

    def test_get_memory_health_score_convenience(self):
        """Test convenience function for memory health score."""
        with patch(
            "services_python.memory_manager._memory_manager.get_memory_health_score"
        ) as mock_get_score:
            mock_get_score.return_value = 85.5

            result = memory_manager_module.get_memory_health_score()

            assert result == 85.5
            mock_get_score.assert_called_once()
