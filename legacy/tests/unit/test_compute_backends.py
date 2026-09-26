"""
Unit tests for compute backends (CUDA, ROCm, MPS, CPU)
"""

from unittest import mock

import pytest

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


def test_cpu_backend_available():
    from worker.src.compute.cpu import CPUBackend

    backend = CPUBackend()
    assert backend.is_available() is True
    assert backend.name == "CPUBackend"


def test_cpu_backend_initialize():
    from worker.src.compute.cpu import CPUBackend

    backend = CPUBackend()
    assert backend.initialize() is True
    assert backend._initialized is True
    info = backend.get_device_info()
    assert "processor" in info
    assert "logical_cores" in info
    backend.cleanup()


@pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
def test_cpu_memory_stats():
    from worker.src.compute.cpu import CPUBackend

    backend = CPUBackend()
    backend.initialize()
    stats = backend.get_memory_stats()
    assert "total_mb" in stats
    assert "available_mb" in stats
    assert stats["total_mb"] > 0
    backend.cleanup()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_cpu_optimize_model():
    from worker.src.compute.cpu import CPUBackend

    backend = CPUBackend()
    backend.initialize()
    model = nn.Linear(10, 10)
    optimized = backend.optimize_model(model)
    assert optimized is not None
    backend.cleanup()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_cpu_create_optimizer():
    from worker.src.compute.cpu import CPUBackend

    backend = CPUBackend()
    backend.initialize()
    model = nn.Linear(10, 10)
    optimizer = backend.create_optimizer(model.parameters(), lr=0.01)
    assert optimizer is not None
    assert optimizer.param_groups[0]["lr"] == 0.01
    backend.cleanup()


def test_cuda_backend_available():
    from worker.src.compute.cuda import CUDABackend

    with mock.patch("torch.cuda.is_available", return_value=True):
        backend = CUDABackend()
        assert backend.is_available() is True


def test_cuda_backend_initialize():
    """Test CUDA backend structure - actual init requires real CUDA."""
    from worker.src.compute.cuda import CUDABackend

    backend = CUDABackend()
    # Verify backend structure
    assert backend.name == "CUDABackend"
    assert hasattr(backend, "device_id")
    assert backend.device_id == 0


def test_mps_backend_available():
    from worker.src.compute.mps import MPSBackend

    with mock.patch("platform.system", return_value="Darwin"):
        with mock.patch(
            "worker.src.compute.mps.backend.torch.backends.mps.is_available", return_value=True
        ):
            backend = MPSBackend()
            assert backend.is_available() is True


def test_mps_backend_initialize():
    """Test MPS backend structure - actual init requires real MPS."""
    from worker.src.compute.mps import MPSBackend

    backend = MPSBackend()
    # Verify backend structure
    assert backend.name == "MPSBackend"
    assert hasattr(backend, "device_id")
    assert backend.device_id == 0


def test_backend_detection():
    from worker.src.compute import detect_backend

    backend = detect_backend()
    assert backend is not None
    assert backend.name in ["CUDABackend", "ROCmBackend", "MPSBackend", "CPUBackend"]


def test_get_backend_by_name():
    from worker.src.compute import get_backend

    cpu = get_backend("cpu")
    assert cpu.name == "CPUBackend"


def test_get_cuda_backend():
    from worker.src.compute import get_backend

    with mock.patch("torch.cuda.is_available", return_value=True):
        cuda = get_backend("cuda")
        assert cuda.name == "CUDABackend"
