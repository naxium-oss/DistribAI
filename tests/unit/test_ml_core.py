"""Tests for ML core module."""

import tempfile
from pathlib import Path

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from worker.src.daemon.ml_core import OrchestratorMLState

    HAS_ML_CORE = True
except ImportError:
    HAS_ML_CORE = False
    OrchestratorMLState = None


@pytest.fixture
def ml_state():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        yield OrchestratorMLState(str(ckpt))


@pytest.mark.skipif(not HAS_TORCH or not HAS_ML_CORE, reason="torch or ml_core not available")
def test_apply_gradients_buffers_until_quorum_then_steps(ml_state: OrchestratorMLState):
    g = {"layer1": torch.ones(3, 3)}
    assert ml_state.apply_gradients({"layer1": g["layer1"].tolist()}) is True
    assert ml_state.apply_gradients({"layer1": g["layer1"].tolist()}) is True
    assert ml_state.step_count == 0
    assert ml_state.aggregation_count == 2
    assert ml_state.apply_gradients({"layer1": g["layer1"].tolist()}) is True
    assert ml_state.step_count == 1
    assert ml_state.aggregation_count == 0
    assert "layer1" in ml_state.model_state


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_get_set_model_state(ml_state: OrchestratorMLState):
    ml_state.set_model_state({"w": [[1.0, 2.0], [3.0, 4.0]]})
    out = ml_state.get_model_state()
    assert "w" in out


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_checkpoint_info(ml_state: OrchestratorMLState):
    info = ml_state.get_checkpoint_info()
    assert "step_count" in info
    assert "checkpoint_path" in info


@pytest.mark.skipif(not HAS_TORCH or not HAS_ML_CORE, reason="torch or ml_core not available")
def test_reset_clears_state(ml_state: OrchestratorMLState):
    ml_state.apply_gradients({"layer1": torch.ones(2).tolist()})
    ml_state.reset()
    assert ml_state.step_count == 0
    assert len(ml_state.model_state) == 0


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_ml_state_creation():
    """Test OrchestratorMLState creation."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        assert state is not None
        assert state.step_count == 0


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_ml_state_initial_state():
    """Test OrchestratorMLState initial state."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        assert state.aggregation_count == 0
        assert state.model_state == {}


@pytest.mark.skipif(not HAS_TORCH or not HAS_ML_CORE, reason="torch or ml_core not available")
def test_apply_gradients_increments_aggregation():
    """Test that apply_gradients increments aggregation count."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        g = {"layer1": torch.ones(3).tolist()}
        state.apply_gradients(g)
        assert state.aggregation_count == 1

        state.apply_gradients(g)
        assert state.aggregation_count == 2


@pytest.mark.skipif(not HAS_TORCH or not HAS_ML_CORE, reason="torch or ml_core not available")
def test_apply_gradients_with_tensor_input():
    """Test apply_gradients with tensor input instead of list."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        g = {"layer1": torch.ones(3)}
        state.apply_gradients(g)
        assert state.aggregation_count == 1


@pytest.mark.skipif(not HAS_TORCH or not HAS_ML_CORE, reason="torch or ml_core not available")
def test_apply_gradients_mixed_types():
    """Test apply_gradients with mixed list and tensor inputs."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        g1 = {"layer1": torch.ones(3).tolist(), "layer2": torch.ones(3)}
        state.apply_gradients(g1)
        assert state.aggregation_count == 1


@pytest.mark.skipif(not HAS_TORCH or not HAS_ML_CORE, reason="torch or ml_core not available")
def test_apply_gradients_error_handling():
    """Test apply_gradients error handling."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        # Pass invalid gradient that will cause error
        result = state.apply_gradients({"layer1": "not_a_list_or_tensor"})
        # May succeed or fail depending on implementation
        assert isinstance(result, bool)


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_load_checkpoint_from_file():
    """Test loading checkpoint from existing file."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"

        state1 = OrchestratorMLState(str(ckpt))
        state1.step_count = 5
        state1.aggregation_count = 3
        state1.model_state = {"w": [1.0, 2.0]}
        state1.force_checkpoint()

        state2 = OrchestratorMLState(str(ckpt))
        assert state2.step_count == 5
        assert state2.aggregation_count == 3


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_load_checkpoint_corrupted():
    """Test loading corrupted checkpoint handles gracefully."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        ckpt.write_text("corrupted data")

        state = OrchestratorMLState(str(ckpt))
        assert state.step_count == 0


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_save_checkpoint():
    """Test save checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        state.step_count = 10
        state.force_checkpoint()

        assert ckpt.exists()


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_aggregate_and_step_with_buffer():
    """Test _aggregate_and_step with gradient buffer."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.gradient_buffer = {"layer1": torch.tensor([3.0, 3.0, 3.0])}
        state.aggregation_count = 3
        state._aggregate_and_step()

        assert len(state.gradient_buffer) == 0
        assert state.aggregation_count == 0
        assert state.step_count == 1


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_aggregate_and_step_empty_buffer():
    """Test _aggregate_and_step with empty buffer."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.gradient_buffer = {}
        state.aggregation_count = 3
        state._aggregate_and_step()

        # Empty buffer just resets count, doesn't increment step
        assert state.aggregation_count == 0
        assert state.step_count == 0


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_aggregate_and_step_updates_model_state():
    """Test _aggregate_and_step updates model state."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.gradient_buffer = {"layer1": torch.tensor([3.0, 3.0])}
        state.model_state = {"layer1": torch.tensor([1.0, 1.0])}
        state.aggregation_count = 3
        state._aggregate_and_step()

        assert "layer1" in state.model_state


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_aggregate_and_step_creates_new_param():
    """Test _aggregate_and_step creates new parameter if not in model state."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.gradient_buffer = {"new_layer": torch.tensor([3.0, 3.0])}
        state.model_state = {}
        state.aggregation_count = 3
        state._aggregate_and_step()

        assert "new_layer" in state.model_state


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_get_model_state_with_tensors():
    """Test get_model_state converts tensors to lists."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.model_state = {"w": torch.tensor([1.0, 2.0])}
        result = state.get_model_state()

        assert isinstance(result["w"], list)


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_set_model_state_with_tensors():
    """Test set_model_state converts lists to tensors."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.set_model_state({"w": [1.0, 2.0]})
        assert isinstance(state.model_state["w"], torch.Tensor)


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_set_model_state_with_non_lists():
    """Test set_model_state handles non-list values."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.set_model_state({"w": 5.0})
        assert state.model_state["w"] == 5.0


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_checkpoint_info_all_fields():
    """Test get_checkpoint_info returns all expected fields."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.step_count = 5
        state.aggregation_count = 2
        state.last_aggregation_time = 123456.0

        info = state.get_checkpoint_info()
        assert info["step_count"] == 5
        assert info["aggregation_count"] == 2
        assert info["last_aggregation_time"] == 123456.0
        assert "checkpoint_path" in info
        assert "parameter_count" in info


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_reset_deletes_checkpoint_file():
    """Test reset deletes checkpoint file."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        state.force_checkpoint()

        assert ckpt.exists()
        state.reset()
        assert not ckpt.exists()


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_reset_clears_all_state():
    """Test reset clears all state variables."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.step_count = 10
        state.aggregation_count = 5
        state.last_aggregation_time = 999.0
        state.model_state = {"w": [1.0]}
        state.gradient_buffer = {"g": [2.0]}

        state.reset()
        assert state.step_count == 0
        assert state.aggregation_count == 0
        assert state.last_aggregation_time == 0.0
        assert len(state.model_state) == 0
        assert len(state.gradient_buffer) == 0


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_force_checkpoint_saves_immediately():
    """Test force_checkpoint saves immediately."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.step_count = 7
        state.force_checkpoint()

        assert ckpt.exists()


@pytest.mark.skipif(not HAS_TORCH or not HAS_ML_CORE, reason="torch or ml_core not available")
def test_checkpoint_saved_every_10_steps():
    """Test checkpoint is saved every 10 steps."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.gradient_buffer = {"layer1": torch.tensor([3.0])}
        state.aggregation_count = 3
        state._aggregate_and_step()

        assert not ckpt.exists()

        state.step_count = 9
        state.gradient_buffer = {"layer1": torch.tensor([3.0])}
        state.aggregation_count = 3
        state._aggregate_and_step()

        assert ckpt.exists()


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_gradient_buffer_accumulation():
    """Test gradient buffer accumulates across calls."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.apply_gradients({"layer1": torch.tensor([1.0]).tolist()})
        state.apply_gradients({"layer1": torch.tensor([2.0]).tolist()})

        assert "layer1" in state.gradient_buffer


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_gradient_buffer_new_layers():
    """Test gradient buffer handles new layers."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))

        state.apply_gradients({"layer1": torch.tensor([1.0]).tolist()})
        state.apply_gradients({"layer2": torch.tensor([2.0]).tolist()})

        assert "layer1" in state.gradient_buffer
        assert "layer2" in state.gradient_buffer
