"""Extended tests for ML core module."""

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


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_ml_state_creation():
    """Test OrchestratorMLState creation."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        assert state is not None
        assert state.step_count == 0


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_ml_state_checkpoint_info():
    """Test checkpoint info retrieval."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        info = state.get_checkpoint_info()
        assert "step_count" in info
        assert "checkpoint_path" in info


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_ml_state_reset():
    """Test state reset."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        state.step_count = 5
        state.reset()
        assert state.step_count == 0


@pytest.mark.skipif(not HAS_ML_CORE or not HAS_TORCH, reason="ml_core or torch not available")
def test_ml_state_apply_gradients():
    """Test applying gradients."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        grads = {"layer1": torch.ones(3, 3).tolist()}
        result = state.apply_gradients(grads)
        assert result is True


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_ml_state_set_get_model_state():
    """Test model state get/set."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        model_data = {"weight": [[1.0, 2.0], [3.0, 4.0]], "bias": [0.5, 0.5]}
        state.set_model_state(model_data)
        retrieved = state.get_model_state()
        assert "weight" in retrieved
        assert "bias" in retrieved


@pytest.mark.skipif(not HAS_ML_CORE, reason="ml_core not available")
def test_ml_state_aggregation_config():
    """Test aggregation configuration."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "ckpt.pt"
        state = OrchestratorMLState(str(ckpt))
        assert hasattr(state, "min_aggregation_count")
        assert state.min_aggregation_count > 0
