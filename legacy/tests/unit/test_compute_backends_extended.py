"""Extended tests for compute backends."""

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


def test_cpu_backend_creation():
    """Test CPUBackend creation."""
    try:
        from worker.src.compute.cpu.backend import CPUBackend
    except ImportError:
        pytest.skip("CPUBackend not available")
        return

    backend = CPUBackend(device_id=0)
    assert backend.device_id == 0
    assert backend.name == "CPUBackend"


def test_cpu_backend_is_available():
    """Test CPUBackend is_available."""
    try:
        from worker.src.compute.cpu.backend import CPUBackend
    except ImportError:
        pytest.skip("CPUBackend not available")
        return

    backend = CPUBackend()
    assert backend.is_available() is True


def test_cpu_backend_initialize():
    """Test CPUBackend initialize."""
    try:
        from worker.src.compute.cpu.backend import CPUBackend
    except ImportError:
        pytest.skip("CPUBackend not available")
        return

    backend = CPUBackend()
    result = backend.initialize()
    assert result is True


def test_cpu_backend_get_device_info():
    """Test CPUBackend get_device_info."""
    try:
        from worker.src.compute.cpu.backend import CPUBackend
    except ImportError:
        pytest.skip("CPUBackend not available")
        return

    backend = CPUBackend()
    info = backend.get_device_info()
    # Check for expected CPU info keys
    assert "processor" in info or "name" in info
    assert "physical_cores" in info or "cores" in info


def test_cuda_backend_import():
    """Test CUDABackend import."""
    try:
        from worker.src.compute.cuda.backend import CUDABackend

        assert CUDABackend is not None
    except ImportError:
        pytest.skip("CUDABackend not available")


def test_mps_backend_import():
    """Test MPSBackend import."""
    try:
        from worker.src.compute.mps.backend import MPSBackend

        assert MPSBackend is not None
    except ImportError:
        pytest.skip("MPSBackend not available")


def test_rocm_backend_import():
    """Test ROCmBackend import."""
    try:
        from worker.src.compute.rocm.backend import ROCmBackend

        assert ROCmBackend is not None
    except ImportError:
        pytest.skip("ROCmBackend not available")
