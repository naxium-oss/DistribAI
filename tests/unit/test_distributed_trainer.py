"""Tests for distributed_trainer module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

# Import the module under test
import services_python.distributed_trainer as dist_trainer


class TestDistributedConfig:
    """Test cases for DistributedConfig dataclass."""

    def test_distributed_config_creation(self):
        """Test creating a DistributedConfig instance."""
        config = dist_trainer.DistributedConfig(
            world_size=4,
            rank=1,
            master_addr="192.168.1.100",
            master_port=29500,
            backend="gloo",
            gradient_sync_steps=100,
        )

        assert config.world_size == 4
        assert config.rank == 1
        assert config.master_addr == "192.168.1.100"
        assert config.master_port == 29500
        assert config.backend == "gloo"
        assert config.gradient_sync_steps == 100

    def test_distributed_config_defaults(self):
        """Test DistributedConfig with default values."""
        config = dist_trainer.DistributedConfig(world_size=2, rank=0, master_addr="localhost")

        assert config.world_size == 2
        assert config.rank == 0
        assert config.master_addr == "localhost"
        assert config.master_port == 29500  # Default
        assert config.backend == "gloo"  # Default
        assert config.gradient_sync_steps == 100  # Default


class TestHandleOOMInAggregation:
    """Test cases for handle_oom_in_aggregation function."""

    @pytest.mark.asyncio
    async def test_handle_oom_success(self):
        """Test successful operation without OOM."""
        operation_func = AsyncMock(return_value="success")

        result = await dist_trainer.handle_oom_in_aggregation("test_op", operation_func)

        assert result == "success"
        operation_func.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_oom_cuda_retry_success(self):
        """Test CUDA OOM handled with cache clear and retry success."""
        operation_func = AsyncMock(
            side_effect=[
                RuntimeError("CUDA out of memory"),
                "success_after_retry",
            ]
        )

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.empty_cache") as mock_empty_cache:
                result = await dist_trainer.handle_oom_in_aggregation("test_op", operation_func)

                assert result == "success_after_retry"
                assert operation_func.call_count == 2
                mock_empty_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_oom_cuda_retry_fails(self):
        """After max_retries, OOM errors propagate."""
        operation_func = AsyncMock(side_effect=RuntimeError("CUDA out of memory"))

        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "services_python.memory_manager.MemoryManager.perform_cleanup",
                new_callable=AsyncMock,
            ) as mock_cleanup:
                with pytest.raises(RuntimeError, match="CUDA out of memory"):
                    await dist_trainer.handle_oom_in_aggregation("test_op", operation_func)

                assert operation_func.call_count == 3
                assert mock_cleanup.call_count == 2

    @pytest.mark.asyncio
    async def test_handle_oom_cuda_not_available(self):
        """OOM is still classified when CUDA device is unavailable (CPU-only runtime)."""
        operation_func = AsyncMock(
            side_effect=[
                MemoryError(),
                "success_after_retry",
            ]
        )

        with patch("torch.cuda.is_available", return_value=False):
            result = await dist_trainer.handle_oom_in_aggregation("test_op", operation_func)

        assert result == "success_after_retry"
        assert operation_func.call_count == 2

    @pytest.mark.asyncio
    async def test_handle_runtime_error(self):
        """Test handling of runtime errors (non-OOM)."""
        operation_func = AsyncMock(side_effect=RuntimeError("Runtime error"))

        with pytest.raises(RuntimeError):
            await dist_trainer.handle_oom_in_aggregation("test_op", operation_func)

        operation_func.assert_called_once()


@pytest.fixture
def mock_db():
    """Mock database fixture."""
    return Mock()


@pytest.fixture
def trainer(mock_db):
    """Create DistributedTrainer instance with mocked dependencies."""
    with patch("services_python.distributed_trainer.get_database", return_value=mock_db):
        return dist_trainer.DistributedTrainer()


class TestDistributedTrainer:
    """Test cases for DistributedTrainer class."""

    def test_trainer_init(self, trainer):
        """Test trainer initialization."""
        assert trainer.db is not None
        assert hasattr(trainer, "active_jobs")
        assert hasattr(trainer, "node_states")

    @pytest.mark.asyncio
    async def test_initialize_distributed_process(self, trainer):
        """Test initializing distributed process."""
        config = dist_trainer.DistributedConfig(world_size=2, rank=0, master_addr="localhost")

        with patch("torch.distributed.init_process_group") as mock_init:
            await trainer.initialize_distributed_process(config)

            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_distributed_process_with_error(self, trainer):
        """Test initializing distributed process with error."""
        config = dist_trainer.DistributedConfig(world_size=2, rank=0, master_addr="localhost")

        with patch("torch.distributed.init_process_group", side_effect=RuntimeError("Init failed")):
            with pytest.raises(RuntimeError):
                await trainer.initialize_distributed_process(config)

    @pytest.mark.asyncio
    async def test_cleanup_distributed_process(self, trainer):
        """Test cleaning up distributed process."""
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.destroy_process_group") as mock_destroy:
                await trainer.cleanup_distributed_process()

                mock_destroy.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_distributed_job(self, trainer):
        """Test starting a distributed job."""
        job = Mock()
        job.job_id = "test_job_123"
        job.gradient_sync_steps = 100
        node_assignments = [("node1", {}), ("node2", {})]

        with patch.object(trainer.db, "assign_node_to_job", new_callable=AsyncMock):
            with patch.object(trainer.db, "update_job_status", new_callable=AsyncMock):
                with patch.object(trainer, "_get_master_address", return_value="localhost"):
                    result = await trainer.start_distributed_job(job, node_assignments)
                    assert result is True
                    assert "test_job_123" in trainer.active_jobs

    @pytest.mark.asyncio
    async def test_coordinate_training(self, trainer):
        """Test coordinating training across nodes."""
        job_id = "test_job_123"
        config = dist_trainer.DistributedConfig(world_size=2, rank=0, master_addr="localhost")

        # No active job: logs warning path only
        await trainer.coordinate_training(job_id, config)

    @pytest.mark.asyncio
    async def test_synchronize_gradients(self, trainer):
        """Test gradient synchronization."""
        gradients = [torch.randn(10, 10) for _ in range(5)]

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.all_reduce") as mock_all_reduce:
                result = await trainer.synchronize_gradients(gradients)

                assert result == gradients
                assert mock_all_reduce.call_count == 5

    @pytest.mark.asyncio
    async def test_synchronize_gradients_with_oom(self, trainer):
        """Test gradient synchronization with OOM handling."""
        gradients = [torch.randn(10, 10)]

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch(
                "torch.distributed.all_reduce", side_effect=torch.cuda.OutOfMemoryError("OOM")
            ):
                # The implementation doesn't currently catch OOM in synchronize_gradients itself
                # but calls all_reduce. If we want it to handle OOM, we should wrap it.
                with pytest.raises(torch.cuda.OutOfMemoryError):
                    await trainer.synchronize_gradients(gradients)

    @pytest.mark.asyncio
    async def test_broadcast_model(self, trainer):
        """Test model broadcasting."""
        model_state = {"layer1": torch.randn(10, 10)}

        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.broadcast") as mock_broadcast:
                result = await trainer.broadcast_model(model_state, src_rank=0)

                assert result == model_state
                mock_broadcast.assert_called()

    @pytest.mark.asyncio
    async def test_collect_node_metrics(self, trainer):
        """Test collecting metrics from all nodes."""
        config = dist_trainer.DistributedConfig(world_size=2, rank=0, master_addr="localhost")

        result = await trainer.collect_node_metrics(config)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_barrier_synchronization(self, trainer):
        """Test barrier synchronization."""
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.barrier") as mock_barrier:
                await trainer.barrier_synchronization()

                mock_barrier.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_distributed_checkpoint(self, trainer):
        """Test saving distributed checkpoint."""
        job_id = "test_job_123"
        checkpoint_data = {"model": torch.randn(10, 10)}

        with patch.object(trainer, "barrier_synchronization") as mock_barrier:
            with patch("torch.save") as mock_save:
                await trainer.save_distributed_checkpoint(job_id, checkpoint_data)

                mock_barrier.assert_called()
                mock_save.assert_called()

    @pytest.mark.asyncio
    async def test_load_distributed_checkpoint(self, trainer):
        """Test loading distributed checkpoint."""
        job_id = "test_job_123"

        with patch("torch.load") as mock_load:
            mock_load.return_value = {"model": torch.randn(10, 10)}

            result = await trainer.load_distributed_checkpoint(job_id)

            assert "model" in result
            mock_load.assert_called()

    def test_get_world_size(self, trainer):
        """Test getting world size."""
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_world_size", return_value=4):
                result = trainer.get_world_size()
                assert result == 4

    def test_get_rank(self, trainer):
        """Test getting current rank."""
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_rank", return_value=1):
                result = trainer.get_rank()
                assert result == 1

    def test_is_distributed_initialized(self, trainer):
        """Test checking if distributed is initialized."""
        with patch("torch.distributed.is_initialized", return_value=True):
            result = trainer.is_distributed_initialized()
            assert result is True

        with patch("torch.distributed.is_initialized", return_value=False):
            result = trainer.is_distributed_initialized()
            assert result is False

    @pytest.mark.asyncio
    async def test_handle_node_failure(self, trainer):
        """Test handling node failure during training."""
        node_id = "node_1"
        job_id = "test_job_123"
        error = "OOM"

        # Mock the job to be in active_jobs
        trainer.active_jobs[job_id] = dist_trainer.DistributedJob(
            job_id=job_id,
            world_size=2,
            master_addr="localhost",
            master_port=29500,
            gradient_sync_steps=100,
        )

        with patch.object(trainer.db, "update_job_status", new_callable=AsyncMock) as mock_update:
            with patch.object(trainer.db, "update_node_heartbeat", new_callable=AsyncMock):
                with patch.object(trainer.db, "add_log", new_callable=AsyncMock):
                    await trainer.handle_node_failure(job_id, node_id, error)

                    mock_update.assert_called()

    @pytest.mark.asyncio
    async def test_dynamic_scaling_add_nodes(self, trainer):
        """Test dynamic scaling - adding nodes."""
        new_world_size = 4
        job_id = "test_job_123"

        with patch.object(trainer, "reinitialize_with_new_nodes") as mock_reinit:
            await trainer.dynamic_scaling_add_nodes(new_world_size, job_id)

            mock_reinit.assert_called_once()

    @pytest.mark.asyncio
    async def test_dynamic_scaling_remove_nodes(self, trainer):
        """Test dynamic scaling - removing nodes."""
        remaining_ranks = [0, 2]
        job_id = "test_job_123"

        with patch.object(trainer, "reinitialize_with_removed_nodes") as mock_reinit:
            await trainer.dynamic_scaling_remove_nodes(remaining_ranks, job_id)

            mock_reinit.assert_called_once()


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_get_local_ip(self):
        """Test getting local IP address."""
        with patch("socket.socket") as mock_socket:
            mock_sock_instance = mock_socket.return_value
            mock_sock_instance.getsockname.return_value = ["192.168.1.100", 12345]

            result = dist_trainer.get_local_ip()

            assert result == "192.168.1.100"

    def test_get_local_ip_fallback(self):
        """Test getting local IP with fallback to localhost."""
        with patch("socket.socket", side_effect=OSError("Network error")):
            result = dist_trainer.get_local_ip()
            assert result == "127.0.0.1"

    def test_validate_config(self):
        """Test distributed configuration validation."""
        # Valid config
        valid_config = dist_trainer.DistributedConfig(
            world_size=4, rank=1, master_addr="192.168.1.100"
        )
        assert dist_trainer.validate_config(valid_config) is True

        # Invalid rank
        invalid_config = dist_trainer.DistributedConfig(
            world_size=2,
            rank=5,  # Rank >= world_size
            master_addr="localhost",
        )
        assert dist_trainer.validate_config(invalid_config) is False

    def test_create_config_from_job(self):
        """Test creating distributed config from job submission."""
        job_submission = Mock()
        job_submission.job_id = "test_job"
        job_submission.assigned_nodes = ["node1", "node2", "node3"]
        job_submission.node_id = "node2"

        config = dist_trainer.create_config_from_job(job_submission)

        assert config.world_size == 3
        assert config.rank == 1  # node2 is second in list
        assert config.master_addr == "node1"  # First node is master

    @pytest.mark.asyncio
    async def test_monitor_distributed_health(self, trainer):
        """Test monitoring distributed training health."""
        config = dist_trainer.DistributedConfig(world_size=2, rank=0, master_addr="localhost")

        with patch.object(trainer, "collect_node_metrics") as mock_collect:
            mock_collect.return_value = [{"status": "healthy"}, {"status": "healthy"}]

            result = await trainer.monitor_distributed_health(config)

            assert result["all_healthy"] is True
            assert isinstance(result["node_status"], list)
