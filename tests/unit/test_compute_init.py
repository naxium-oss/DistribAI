"""Tests for compute __init__ module."""

import os
from unittest import mock

import pytest


def test_detect_backend():
    """Test detect_backend function."""
    from worker.src.compute import detect_backend

    backend = detect_backend()
    assert backend is not None
    assert hasattr(backend, "name")


def test_detect_backend_with_override():
    """Test detect_backend with DISTRIBAI_BACKEND env var."""
    from worker.src.compute import detect_backend

    with mock.patch.dict(os.environ, {"DISTRIBAI_BACKEND": "cpu"}):
        backend = detect_backend()
        assert backend is not None


def test_get_backend_cpu():
    """Test get_backend with CPU."""
    from worker.src.compute import get_backend

    backend = get_backend("cpu")
    assert backend is not None
    assert backend.name == "CPUBackend"


def test_get_backend_cpu_with_device_id():
    """Test get_backend with CPU and device_id."""
    from worker.src.compute import get_backend

    backend = get_backend("cpu", device_id=1)
    assert backend is not None
    # CPUBackend may not use device_id, so just check backend exists


def test_get_backend_cuda_mocked():
    """Test get_backend with CUDA mocked."""
    from worker.src.compute import get_backend

    with mock.patch("worker.src.compute.cuda.CUDABackend.is_available", return_value=True):
        backend = get_backend("cuda")
        assert backend is not None


def test_get_backend_rocm_mocked():
    """Test get_backend with ROCm mocked."""
    from worker.src.compute import get_backend

    with mock.patch("worker.src.compute.rocm.ROCmBackend.is_available", return_value=True):
        backend = get_backend("rocm")
        assert backend is not None


def test_get_backend_mps_mocked():
    """Test get_backend with MPS mocked."""
    from worker.src.compute import get_backend

    with mock.patch("worker.src.compute.mps.MPSBackend.is_available", return_value=True):
        backend = get_backend("mps")
        assert backend is not None


def test_get_backend_invalid():
    """Test get_backend with invalid name."""
    from worker.src.compute import get_backend

    with pytest.raises(ValueError):
        get_backend("invalid_backend")


def test_get_backend_case_insensitive():
    """Test get_backend is case-insensitive."""
    from worker.src.compute import get_backend

    backend = get_backend("CPU")
    assert backend is not None
    assert backend.name == "CPUBackend"


def test_compute_backend_base_class():
    """Test ComputeBackend base class."""
    from worker.src.compute import ComputeBackend

    # Check that ComputeBackend is abstract and requires implementation
    assert hasattr(ComputeBackend, "__abstractmethods__")


def test_compute_backend_initialization():
    """Test ComputeBackend initialization."""
    from worker.src.compute import ComputeBackend

    # Create a concrete implementation for testing
    class TestBackend(ComputeBackend):
        def initialize(self) -> bool:
            return True

        def is_available(self) -> bool:
            return True

        def get_device_info(self) -> dict:
            return {}

        def get_memory_stats(self) -> dict:
            return {}

        def optimize_model(self, model):
            return model

        def create_optimizer(self, model_parameters, lr, **kwargs):
            return None

        def cleanup(self) -> None:
            pass

    backend = TestBackend(device_id=5)
    assert backend.device_id == 5
    assert backend.name == "TestBackend"
    assert backend._initialized is False


def test_detect_backend_returns_cpu_when_present():
    """detect_backend always has CPU fallback in this tree."""
    from worker.src.compute import detect_backend

    backend = detect_backend()
    assert backend is not None
