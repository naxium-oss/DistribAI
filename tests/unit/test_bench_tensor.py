"""Tests for tensor benchmark module."""

import os
from unittest import mock

import pytest

try:
    from worker.src.benchmark import bench_tensor

    HAS_BENCH_TENSOR = True
except ImportError:
    HAS_BENCH_TENSOR = False
    bench_tensor = None


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_emit_outputs_json(capsys):
    """Test emit function outputs JSON to stdout."""
    import json

    test_data = {"type": "test", "value": 42}
    bench_tensor.emit(test_data)
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip()) == test_data


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_log_score():
    """Test log_score function."""
    result = bench_tensor.log_score(10.0, 1.0, 100.0)
    assert isinstance(result, float)
    assert 0 <= result <= 100


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_log_score_zero():
    """Test log_score with zero value."""
    result = bench_tensor.log_score(0.0, 1.0, 100.0)
    assert result == 0.0


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_log_score_invalid_floor():
    """Test log_score with invalid floor."""
    result = bench_tensor.log_score(10.0, 0.0, 100.0)
    assert result == 0.0


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_log_score_floor_greater_than_ceil():
    """Test log_score with floor greater than ceil."""
    result = bench_tensor.log_score(10.0, 100.0, 50.0)
    assert result == 0.0


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_log_score_at_ceil():
    """Test log_score at ceiling value."""
    result = bench_tensor.log_score(100.0, 1.0, 100.0)
    assert result == 100.0


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_log_score_exceeds_ceil():
    """Test log_score capped at 100."""
    result = bench_tensor.log_score(200.0, 1.0, 100.0)
    assert result == 100.0


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_mean_helper():
    """Test _mean helper function."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = bench_tensor._mean(values)
    assert result == 3.0


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_mean_empty():
    """Test _mean with empty list."""
    result = bench_tensor._mean([])
    assert result == 0.0


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_detect_throttle_insufficient_data():
    """Test _detect_throttle with insufficient data points."""
    result = bench_tensor._detect_throttle([1.0, 2.0, 3.0])
    assert result is False


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_detect_throttle_no_throttle():
    """Test _detect_throttle with no throttling."""
    throughputs = [1.0] * 60
    result = bench_tensor._detect_throttle(throughputs)
    assert result is False


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_detect_throttle_detected():
    """Test _detect_throttle detects throttling."""
    throughputs = [1.0] * 20 + [0.7] * 40
    result = bench_tensor._detect_throttle(throughputs)
    assert result is True


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_count_params():
    """Test _count_params function."""
    try:
        import torch.nn as nn

        model = nn.Linear(10, 5)
        result = bench_tensor._count_params(model)
        assert result > 0
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_build_model_one_param():
    """Test _build_model with 1 parameter."""
    try:
        model = bench_tensor._build_model(1)
        assert model is not None
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_build_model_small():
    """Test _build_model with small target."""
    try:
        model = bench_tensor._build_model(10)
        assert model is not None
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_build_model_medium():
    """Test _build_model with medium target."""
    try:
        model = bench_tensor._build_model(100)
        assert model is not None
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_build_model_large():
    """Test _build_model with large target."""
    try:
        model = bench_tensor._build_model(1000)
        assert model is not None
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_make_batch():
    """Test _make_batch function."""
    try:
        import torch

        device = torch.device("cpu")
        x, y = bench_tensor._make_batch(device, 0)
        assert x.shape == (1, 1)
        assert y.shape == (1, 1)
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_main_no_torch():
    """Test main function without torch."""
    with mock.patch("worker.src.benchmark.bench_tensor._HAS_TORCH", False):
        with mock.patch("worker.src.benchmark.bench_tensor.emit") as mock_emit:
            result = bench_tensor.main()
            assert result is None
            assert mock_emit.called


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_main_with_torch_cpu():
    """Test main function with torch on CPU."""
    try:
        import torch

        with mock.patch("worker.src.benchmark.bench_tensor._HAS_TORCH", True):
            with mock.patch("torch.cuda.is_available", return_value=False):
                with mock.patch.object(torch.backends, "mps", create=True):
                    with mock.patch("worker.src.benchmark.bench_tensor._build_model"):
                        with mock.patch(
                            "worker.src.benchmark.bench_tensor._count_params", return_value=10
                        ):
                            with mock.patch(
                                "worker.src.benchmark.bench_tensor._train_model"
                            ) as mock_train:
                                mock_train.return_value = {
                                    "steps_per_sec": 100.0,
                                    "ms_per_step": 10.0,
                                    "final_loss": 0.5,
                                    "throttled": False,
                                }
                                with mock.patch("worker.src.benchmark.bench_tensor.emit"):
                                    result = bench_tensor.main()
                                    assert result is not None
                                    assert "score" in result
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_main_with_cuda():
    """Test main function with CUDA device."""
    try:
        with mock.patch("worker.src.benchmark.bench_tensor._HAS_TORCH", True):
            with mock.patch("torch.cuda.is_available", return_value=True):
                with mock.patch("worker.src.benchmark.bench_tensor._build_model"):
                    with mock.patch(
                        "worker.src.benchmark.bench_tensor._count_params", return_value=10
                    ):
                        with mock.patch(
                            "worker.src.benchmark.bench_tensor._train_model"
                        ) as mock_train:
                            mock_train.return_value = {
                                "steps_per_sec": 1000.0,
                                "ms_per_step": 1.0,
                                "final_loss": 0.3,
                                "throttled": False,
                            }
                            with mock.patch("worker.src.benchmark.bench_tensor.emit"):
                                result = bench_tensor.main()
                                assert result is not None
                                assert result["gpu_used"] is True
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_train_model():
    """Test _train_model function."""
    try:
        import torch
        import torch.nn as nn

        device = torch.device("cpu")
        model = nn.Linear(1, 1)
        result = bench_tensor._train_model(model, device, 10, 5, 1)
        assert "steps_per_sec" in result
        assert "ms_per_step" in result
        assert "final_loss" in result
        assert "throttled" in result
    except ImportError:
        pytest.skip("torch not available")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_constants():
    """Test module constants."""
    assert hasattr(bench_tensor, "_FLOOR")
    assert hasattr(bench_tensor, "_CEIL")
    assert hasattr(bench_tensor, "_STEPS")
    assert hasattr(bench_tensor, "_WARMUP")
    assert hasattr(bench_tensor, "_PARAM_TARGETS")


@pytest.mark.skipif(not HAS_BENCH_TENSOR, reason="bench_tensor not available")
def test_env_vars():
    """Test environment variables affect constants."""
    original_floor = os.environ.get("SCORE_FLOOR_SPS")
    original_ceil = os.environ.get("SCORE_CEIL_SPS")
    original_steps = os.environ.get("TENSOR_STEPS")
    original_warmup = os.environ.get("TENSOR_WARMUP")

    try:
        os.environ["SCORE_FLOOR_SPS"] = "2.0"
        os.environ["SCORE_CEIL_SPS"] = "100000.0"
        os.environ["TENSOR_STEPS"] = "100"
        os.environ["TENSOR_WARMUP"] = "10"

        import importlib

        import worker.src.benchmark.bench_tensor as bench_tensor

        importlib.reload(bench_tensor)

        assert bench_tensor._FLOOR == 2.0
        assert bench_tensor._CEIL == 100000.0
        assert bench_tensor._STEPS == 100
        assert bench_tensor._WARMUP == 10
    finally:
        if original_floor:
            os.environ["SCORE_FLOOR_SPS"] = original_floor
        else:
            os.environ.pop("SCORE_FLOOR_SPS", None)
        if original_ceil:
            os.environ["SCORE_CEIL_SPS"] = original_ceil
        else:
            os.environ.pop("SCORE_CEIL_SPS", None)
        if original_steps:
            os.environ["TENSOR_STEPS"] = original_steps
        else:
            os.environ.pop("TENSOR_STEPS", None)
        if original_warmup:
            os.environ["TENSOR_WARMUP"] = original_warmup
        else:
            os.environ.pop("TENSOR_WARMUP", None)
