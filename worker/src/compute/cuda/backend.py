"""
CUDA Compute Backend for DistribAI

Implements CUDA-based computation for NVIDIA GPUs.
Optimized for PyTorch with cuDNN, TensorRT, and Flash Attention support.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.backends.cudnn as cudnn

try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

from .. import ComputeBackend

logger = logging.getLogger(__name__)


class CUDABackend(ComputeBackend):
    """
    NVIDIA GPU compute backend using CUDA.

    Provides optimized training and inference on NVIDIA GPUs with
    cuDNN acceleration, memory management, and torch.compile support.

    Attributes:
        device_id: CUDA device ID
        device: PyTorch device object
        _nvml_initialized: Whether NVML is initialized
        _handle: NVML device handle

    Example:
        backend = CUDABackend(device_id=0)
        backend.initialize()
        info = backend.get_device_info()
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize the CUDA backend.

        Args:
            device_id: CUDA device ID (default: 0)

        Example:
            >>> backend = CUDABackend(device_id=0)
        """
        super().__init__(device_id)
        self.device: torch.device | None = None
        self._nvml_initialized = False
        self._handle = None

    def is_available(self) -> bool:
        """
        Check if CUDA is available.

        Returns:
            True if CUDA is available on the system

        Example:
            >>> backend.is_available()
            True
        """
        return torch.cuda.is_available()

    def initialize(self) -> bool:
        """
        Initialize the CUDA backend.

        Sets up CUDA device, cuDNN, and NVML for memory monitoring.
        Enables TF32 for faster training on Ampere+ GPUs.

        Returns:
            True if initialization successful

        Example:
            >>> success = backend.initialize()
        """
        if not self.is_available():
            logger.error("CUDA is not available on this system")
            return False
        try:
            torch.cuda.set_device(self.device_id)
            self.device = torch.device(f"cuda:{self.device_id}")
            cudnn.benchmark = True
            cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if PYNVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    self._nvml_initialized = True
                    self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                    logger.info(f"NVML initialized for GPU {self.device_id}")
                except Exception as e:
                    logger.warning(f"NVML initialization failed: {e}")
            self._initialized = True
            logger.info(f"CUDA backend initialized on {self.get_device_info()['name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CUDA: {e}")
            return False

    def get_device_info(self) -> dict[str, Any]:
        """
        Get GPU device information.

        Returns:
            Dictionary with GPU specifications and feature support

        Example:
            >>> info = backend.get_device_info()
            >>> print(f"GPU: {info['name']}, Memory: {info['total_memory_mb']} MB")
        """
        if not self._initialized:
            return {"error": "Backend not initialized"}
        props = torch.cuda.get_device_properties(self.device_id)
        capability = f"{props.major}.{props.minor}"
        arch_names = {
            (3, 0): "Kepler",
            (3, 5): "Kepler",
            (3, 7): "Kepler",
            (5, 0): "Maxwell",
            (5, 2): "Maxwell",
            (5, 3): "Maxwell",
            (6, 0): "Pascal",
            (6, 1): "Pascal",
            (6, 2): "Pascal",
            (7, 0): "Volta",
            (7, 5): "Turing",
            (8, 0): "Ampere",
            (8, 6): "Ampere",
            (8, 9): "Ada Lovelace",
            (9, 0): "Hopper",
        }
        arch = arch_names.get((props.major, props.minor), "Unknown")
        return {
            "name": torch.cuda.get_device_name(self.device_id),
            "id": self.device_id,
            "total_memory_mb": props.total_memory // (1024 * 1024),
            "multi_processor_count": props.multi_processor_count,
            "compute_capability": capability,
            "architecture": arch,
            "supports_fp16": props.major >= 7,
            "supports_bf16": props.major >= 8,
            "supports_tensor_cores": props.major >= 7,
            "supports_flash_attention": props.major >= 7,
        }

    def get_memory_stats(self) -> dict[str, int]:
        """
        Get GPU memory statistics.

        Returns:
            Dictionary with GPU memory usage in MB

        Example:
            >>> stats = backend.get_memory_stats()
            >>> print(f"Free: {stats['free_mb']} MB")
        """
        if not self._initialized:
            return {"error": "Backend not initialized"}
        if self._nvml_initialized and self._handle:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return {
                    "total_mb": mem_info.total // (1024 * 1024),
                    "free_mb": mem_info.free // (1024 * 1024),
                    "used_mb": mem_info.used // (1024 * 1024),
                    "reserved_mb": 0,
                }
            except (RuntimeError, OSError):
                pass
        torch_stats = torch.cuda.memory_stats(self.device_id)
        return {
            "total_mb": torch.cuda.get_device_properties(self.device_id).total_memory
            // (1024 * 1024),
            "allocated_mb": torch_stats.get("allocated_bytes.all.current", 0) // (1024 * 1024),
            "reserved_mb": torch_stats.get("reserved_bytes.all.current", 0) // (1024 * 1024),
            "free_mb": 0,
        }

    def optimize_model(self, model: Any) -> Any:
        """
        Optimize a model for GPU execution.

        Moves model to GPU and applies torch.compile for faster execution.

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model on GPU

        Raises:
            RuntimeError: If backend not initialized

        Example:
            >>> optimized = backend.optimize_model(model)
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized")
        model = model.to(self.device)
        compile_enabled = os.getenv("DISTRIBAI_TORCH_COMPILE", "false").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if compile_enabled and hasattr(torch, "compile") and hasattr(model, "forward"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        return model

    def create_optimizer(self, model_parameters: Any, lr: float, **kwargs) -> Any:
        """
        Create a GPU-optimized optimizer.

        Args:
            model_parameters: Model parameters to optimize
            lr: Learning rate
            **kwargs: Additional optimizer arguments

        Returns:
            Configured optimizer with fused operations if available

        Example:
            >>> optimizer = backend.create_optimizer(model.parameters(), lr=0.001)
        """
        import torch.optim as optim

        fused_available = hasattr(torch.optim.AdamW, "fused") if hasattr(optim, "AdamW") else False
        optimizer_class = kwargs.pop("optimizer_class", optim.AdamW)
        defaults = {
            "lr": lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
        if fused_available and "fused" not in kwargs:
            defaults["fused"] = True
            logger.info("Using fused AdamW optimizer")
        defaults.update(kwargs)
        return optimizer_class(model_parameters, **defaults)

    def get_stream(self) -> torch.cuda.Stream:
        """
        Get a CUDA stream for asynchronous operations.

        Returns:
            CUDA stream for the device

        Raises:
            RuntimeError: If backend not initialized

        Example:
            >>> stream = backend.get_stream()
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized")
        return torch.cuda.Stream(device=self.device)

    def synchronize(self):
        """
        Synchronize all CUDA operations on the device.

        Waits for all GPU operations to complete.

        Example:
            >>> backend.synchronize()
        """
        if self._initialized:
            torch.cuda.synchronize(self.device_id)

    def cleanup(self):
        """
        Clean up CUDA backend resources.

        Clears GPU cache and shuts down NVML.

        Example:
            >>> backend.cleanup()
        """
        if self._initialized:
            torch.cuda.empty_cache()
            if self._nvml_initialized:
                try:
                    pynvml.nvmlShutdown()
                except (RuntimeError, OSError):
                    pass
            self._initialized = False
            logger.info("CUDA backend cleaned up")
