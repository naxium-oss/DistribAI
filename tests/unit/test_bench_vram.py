"""Tests for VRAM benchmark module."""

import os
import sys
from unittest import mock

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/tests/", 1)[0])

try:
    from worker.src.benchmark.bench_vram import (
        _CEIL,
        _FLOOR,
        _build_flat_model,
        _max_raw_alloc_gb,
        _try_train,
        emit,
        log_score,
        main,
    )

    HAS_BENCH_VRAM = True
except ImportError:
    HAS_BENCH_VRAM = False
    # Define dummy functions for test collection
    emit = None
    _max_raw_alloc_gb = None
    _try_train = None
    _build_flat_model = None
    log_score = None
    main = None
    _FLOOR = 0.5
    _CEIL = 128.0


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_emit(capsys):
    """Test emit function outputs JSON."""
    import json

    test_data = {"type": "test", "value": 42}
    emit(test_data)
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip()) == test_data


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_log_score_basic():
    """Test log_score function."""
    result = log_score(10.0, 1.0, 100.0)
    assert 0.0 <= result <= 100.0


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_log_score_zero_value():
    """Test log_score with zero value."""
    result = log_score(0.0, 1.0, 100.0)
    assert result == 0.0


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_log_score_at_floor():
    """Test log_score at floor value."""
    result = log_score(0.5, 0.5, 128.0)
    assert result == 0.0


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_log_score_at_ceil():
    """Test log_score at ceiling value."""
    result = log_score(128.0, 0.5, 128.0)
    assert result == 100.0


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_log_score_exceeds_ceil():
    """Test log_score capped at 100."""
    result = log_score(256.0, 0.5, 128.0)
    assert result == 100.0


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_log_score_invalid_floor():
    """Test log_score with invalid floor."""
    result = log_score(10.0, 0.0, 100.0)
    assert result == 0.0


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_log_score_floor_greater_than_ceil():
    """Test log_score with floor greater than ceil."""
    result = log_score(10.0, 100.0, 50.0)
    assert result == 0.0


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_build_flat_model_fp32():
    """Test model building with fp32."""
    model = _build_flat_model(1.0, torch.float32)
    assert model is not None
    assert hasattr(model, "parameters")


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_build_flat_model_fp16():
    """Test model building with fp16."""
    model = _build_flat_model(1.0, torch.float16)
    assert model is not None


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_build_flat_model_bf16():
    """Test model building with bf16."""
    model = _build_flat_model(1.0, torch.bfloat16)
    assert model is not None


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_build_flat_model_no_torch():
    """Test model building without torch raises ImportError."""
    with mock.patch("worker.src.benchmark.bench_vram._HAS_TORCH", False):
        with pytest.raises(ImportError):
            _build_flat_model(1.0, torch.float32 if torch else None)


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_try_train_oom_error():
    """Test _try_train handles OOM error."""
    with mock.patch("torch.cuda.is_available", return_value=True):
        with mock.patch("worker.src.benchmark.bench_vram._build_flat_model") as mock_build:
            mock_build.side_effect = torch.cuda.OutOfMemoryError()
            result = _try_train(0, torch.float32, 0.001)
            assert result is False


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_try_train_runtime_error():
    """Test _try_train handles runtime error."""
    with mock.patch("torch.cuda.is_available", return_value=True):
        with mock.patch("worker.src.benchmark.bench_vram._build_flat_model") as mock_build:
            mock_build.side_effect = RuntimeError()
            result = _try_train(0, torch.float32, 0.001)
            assert result is False


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_try_train_type_error():
    """Test _try_train handles type error."""
    with mock.patch("torch.cuda.is_available", return_value=True):
        with mock.patch("worker.src.benchmark.bench_vram._build_flat_model") as mock_build:
            mock_build.side_effect = TypeError()
            result = _try_train(0, torch.float32, 0.001)
            assert result is False


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_max_raw_alloc_gb_success():
    """Test _max_raw_alloc_gb successful allocation."""
    mock_cuda = mock.Mock()
    mock_cuda.type = "cuda"
    with mock.patch("torch.cuda.is_available", return_value=True):
        with mock.patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value = mock.Mock(total_memory=8 * 2**30)
            with mock.patch("torch.empty") as mock_empty:
                mock_empty.return_value = mock.MagicMock()
                result = _max_raw_alloc_gb(mock_cuda)
                assert isinstance(result, float)
                assert result > 0


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_max_raw_alloc_gb_oom():
    """Test _max_raw_alloc_gb handles OOM."""
    mock_cuda = mock.Mock()
    mock_cuda.type = "cuda"
    with mock.patch("torch.cuda.is_available", return_value=True):
        with mock.patch("torch.cuda.get_device_properties") as mock_props:
            mock_props.return_value = mock.Mock(total_memory=8 * 2**30)
            with mock.patch("torch.empty", side_effect=torch.cuda.OutOfMemoryError()):
                result = _max_raw_alloc_gb(mock_cuda)
                assert result == 0.0


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_main_no_cuda():
    """Test main function without CUDA."""
    with mock.patch("worker.src.benchmark.bench_vram._HAS_CUDA", False):
        with mock.patch("worker.src.benchmark.bench_vram.emit") as mock_emit:
            result = main()
            assert result is None
            assert mock_emit.called


@pytest.mark.skipif(not HAS_BENCH_VRAM or not HAS_TORCH, reason="bench_vram or torch not available")
def test_main_with_cuda():
    """Test main function with CUDA."""
    with mock.patch("worker.src.benchmark.bench_vram._HAS_CUDA", True):
        with mock.patch("worker.src.benchmark.bench_vram._HAS_ANY_GPU", True):
            with mock.patch("worker.src.benchmark.bench_vram._HAS_TORCH", True):
                with mock.patch("torch.cuda.is_available", return_value=True):
                    with mock.patch("torch.cuda.get_device_properties") as mock_props:
                        mock_props.return_value = mock.Mock(total_memory=8 * 2**30)
                        with mock.patch(
                            "worker.src.benchmark.bench_vram._max_raw_alloc_gb", return_value=4.0
                        ):
                            with mock.patch(
                                "worker.src.benchmark.bench_vram._try_train", return_value=True
                            ):
                                with mock.patch("torch.cuda.is_bf16_supported", return_value=False):
                                    with mock.patch("worker.src.benchmark.bench_vram.emit"):
                                        result = main()
                                        assert result is not None
                                        assert "score" in result


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_floor_ceil_constants():
    """Test _FLOOR and _CEIL constants."""
    assert _FLOOR > 0
    assert _CEIL > _FLOOR


@pytest.mark.skipif(not HAS_BENCH_VRAM, reason="bench_vram not available")
def test_floor_ceil_env_vars():
    """Test environment variables affect _FLOOR and _CEIL."""
    original_floor = os.environ.get("SCORE_FLOOR_GB")
    original_ceil = os.environ.get("SCORE_CEIL_GB")

    try:
        os.environ["SCORE_FLOOR_GB"] = "1.0"
        os.environ["SCORE_CEIL_GB"] = "256.0"
        # Reload module to pick up env vars
        import importlib

        import worker.src.benchmark.bench_vram as bench_vram

        importlib.reload(bench_vram)

        assert bench_vram._FLOOR == 1.0
        assert bench_vram._CEIL == 256.0
    finally:
        if original_floor:
            os.environ["SCORE_FLOOR_GB"] = original_floor
        else:
            os.environ.pop("SCORE_FLOOR_GB", None)
        if original_ceil:
            os.environ["SCORE_CEIL_GB"] = original_ceil
        else:
            os.environ.pop("SCORE_CEIL_GB", None)
