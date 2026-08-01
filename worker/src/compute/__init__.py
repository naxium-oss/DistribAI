"""
DistribAI Hardware-Specific Compute Backends

Provides optimized training backends for:
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- MPS (Apple Silicon)
- CPU (Generic/Intel/AMD)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ComputeBackend(ABC):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.name = self.__class__.__name__
        self._initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_device_info(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_memory_stats(self) -> dict[str, int]:
        pass

    @abstractmethod
    def optimize_model(self, model: Any) -> Any:
        pass

    @abstractmethod
    def create_optimizer(self, model_parameters: Any, lr: float, **kwargs) -> Any:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass


def detect_backend() -> ComputeBackend | None:
    """
    Auto-detect the best available compute backend.
    Order: CUDA -> ROCm -> MPS -> CPU
    """
    backends = []
    try:
        from .cuda.backend import CUDABackend

        backends.append(CUDABackend)
    except ImportError:
        pass
    try:
        from .rocm.backend import ROCmBackend

        backends.append(ROCmBackend)
    except ImportError:
        pass
    try:
        from .mps.backend import MPSBackend

        backends.append(MPSBackend)
    except ImportError:
        pass
    try:
        from .cpu.backend import CPUBackend

        backends.append(CPUBackend)
    except ImportError:
        pass
    backend_override = os.getenv("DISTRIBAI_BACKEND", "").lower()
    for backend_class in backends:
        backend_name = backend_class.__name__.lower().replace("backend", "")
        if backend_override and backend_override != backend_name:
            continue
        try:
            backend = backend_class()
            if backend.is_available():
                if backend.initialize():
                    return backend
        except (RuntimeError, OSError, ImportError):
            continue
    return None


def get_backend(name: str, device_id: int = 0) -> ComputeBackend:
    """
    Get a specific backend by name.
    Args:
        name: One of 'cuda', 'rocm', 'mps', 'cpu'
        device_id: Device index for multi-GPU systems
    Returns:
        ComputeBackend instance
    """
    name = name.lower()
    if name == "cuda":
        from .cuda.backend import CUDABackend

        return CUDABackend(device_id)
    elif name == "rocm":
        from .rocm.backend import ROCmBackend

        return ROCmBackend(device_id)
    elif name == "mps":
        from .mps.backend import MPSBackend

        return MPSBackend(device_id)
    elif name == "cpu":
        from .cpu.backend import CPUBackend

        return CPUBackend(device_id)
    else:
        raise ValueError(f"Unknown backend: {name}")
