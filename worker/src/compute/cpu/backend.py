"""
CPU Compute Backend for DistribAI

Implements CPU-based computation with Intel/AMD optimizations.
Supports Intel Extension for PyTorch (IPEX), OpenVINO, and MKL.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import Any

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class CPUBackend:
    """
    CPU compute backend with Intel/AMD optimizations.

    Supports Intel Extension for PyTorch (IPEX), OpenVINO, and MKL-DNN
    for optimized CPU-based machine learning workloads.

    Attributes:
        device_id: CPU device ID (always 0 for CPU)
        name: Backend name
        _initialized: Whether backend is initialized
        _ipex_available: Whether IPEX is available
        _openvino_available: Whether OpenVINO is available
        _mkl_available: Whether MKL-DNN is available
        _num_threads: Number of OMP threads

    Example:
        backend = CPUBackend()
        backend.initialize()
        info = backend.get_device_info()
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize the CPU backend.

        Args:
            device_id: CPU device ID (always 0 for CPU)

        Example:
            >>> backend = CPUBackend()
        """
        self.device_id = 0
        self.name = "CPUBackend"
        self._initialized = False
        self._ipex_available = False
        self._openvino_available = False
        self._mkl_available = False
        self._num_threads = int(os.getenv("OMP_NUM_THREADS", 0)) or None

    def is_available(self) -> bool:
        """
        Check if CPU is available.

        Returns:
            Always True for CPU backend

        Example:
            >>> backend.is_available()
            True
        """
        return True

    def initialize(self) -> bool:
        """
        Initialize the CPU backend with optimizations.

        Sets up MKL-DNN, IPEX, and OpenVINO if available.
        Configures threading and affinity settings.

        Returns:
            True if initialization successful

        Example:
            >>> success = backend.initialize()
            >>> print(f"Initialized: {success}")
        """
        try:
            if not HAS_TORCH:
                logger.warning("PyTorch not available, CPU backend limited")
                self._initialized = True
                return True
            if self._num_threads:
                torch.set_num_threads(self._num_threads)
                logger.info(f"Set PyTorch threads to {self._num_threads}")
            try:
                import torch.backends.mkldnn

                torch.backends.mkldnn.enabled = True
                self._mkl_available = True
                logger.info("MKL-DNN enabled")
            except (ImportError, AttributeError):
                pass
            try:
                import intel_extension_for_pytorch as ipex

                self._ipex_available = True
                logger.info("Intel Extension for PyTorch (IPEX) loaded")
            except ImportError:
                pass
            try:
                import openvino

                self._openvino_available = True
                logger.info("OpenVINO available")
            except ImportError:
                pass
            if "KMP_AFFINITY" not in os.environ:
                os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
            if "KMP_BLOCKTIME" not in os.environ:
                os.environ["KMP_BLOCKTIME"] = "0"
            self._initialized = True
            logger.info(f"CPU backend initialized: {self.get_device_info()['processor']}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CPU backend: {e}")
            self._initialized = True
            return True

    def get_device_info(self) -> dict[str, Any]:
        """
        Get CPU device information.

        Returns:
            Dictionary with CPU specifications and feature availability

        Example:
            >>> info = backend.get_device_info()
            >>> print(f"Cores: {info['physical_cores']}")
        """
        cpu_info = {
            "processor": platform.processor() or "Unknown",
            "machine": platform.machine(),
            "system": platform.system(),
            "physical_cores": psutil.cpu_count(logical=False) if HAS_PSUTIL else 0,
            "logical_cores": psutil.cpu_count(logical=True) if HAS_PSUTIL else 0,
            "frequency_mhz": 0,
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 1)
            if HAS_PSUTIL
            else 0,
            "mkl_available": self._mkl_available,
            "ipex_available": self._ipex_available,
            "openvino_available": self._openvino_available,
            "pytorch_threads": torch.get_num_threads() if HAS_TORCH else 0,
            "supports_avx2": False,
            "supports_avx512": False,
        }
        if HAS_PSUTIL:
            try:
                freq = psutil.cpu_freq()
                if freq:
                    cpu_info["frequency_mhz"] = int(freq.max or freq.current)
            except (OSError, ValueError):
                pass
        # Check CPU instruction set support (Linux-specific /proc/cpuinfo)
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    cpuinfo = f.read()
                    cpu_info["supports_avx2"] = "avx2" in cpuinfo.lower()
                    cpu_info["supports_avx512"] = "avx512" in cpuinfo.lower()
            except OSError:
                pass
        else:
            # For macOS/Windows, use platform-specific detection
            try:
                # Try to detect AVX support through CPUID via subprocess
                result = subprocess.run(
                    ["sysctl", "hw.optional.avx2_0"]
                    if platform.system() == "Darwin"
                    else ["wmic", "cpu", "get", "name"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    cpu_info["supports_avx2"] = "avx2" in result.stdout.lower()
                    cpu_info["supports_avx512"] = "avx512" in result.stdout.lower()
            except (OSError, subprocess.TimeoutExpired, ValueError):
                # Default to unknown if detection fails
                cpu_info["supports_avx2"] = False
                cpu_info["supports_avx512"] = False
        return cpu_info

    def get_memory_stats(self) -> dict[str, int]:
        """
        Get system memory statistics.

        Returns:
            Dictionary with memory usage in MB

        Example:
            >>> stats = backend.get_memory_stats()
            >>> print(f"Available: {stats['available_mb']} MB")
        """
        if not HAS_PSUTIL:
            return {"total_mb": 0, "available_mb": 0, "percent_used": 0}
        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total // (1024 * 1024),
            "available_mb": mem.available // (1024 * 1024),
            "used_mb": mem.used // (1024 * 1024),
            "free_mb": mem.free // (1024 * 1024),
            "cached_mb": getattr(mem, "cached", 0) // (1024 * 1024),
            "buffers_mb": getattr(mem, "buffers", 0) // (1024 * 1024),
        }

    def optimize_model(self, model: Any) -> Any:
        """
        Optimize a model for CPU execution.

        Applies IPEX optimization and channels_last memory format.

        Args:
            model: PyTorch model to optimize

        Returns:
            Optimized model

        Raises:
            RuntimeError: If backend not initialized

        Example:
            >>> optimized = backend.optimize_model(model)
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized")
        model = model.cpu()
        if self._ipex_available:
            try:
                import intel_extension_for_pytorch as ipex

                model = ipex.optimize(model)
                logger.info("Applied Intel IPEX optimization")
            except Exception as e:
                logger.warning(f"IPEX optimization failed: {e}")
        if HAS_TORCH and hasattr(model, "to"):
            model = model.to(memory_format=torch.channels_last)
        return model

    def create_optimizer(self, model_parameters: Any, lr: float, **kwargs):
        """
        Create an optimizer for training.

        Args:
            model_parameters: Model parameters to optimize
            lr: Learning rate
            **kwargs: Additional optimizer arguments

        Returns:
            Configured optimizer instance

        Example:
            >>> optimizer = backend.create_optimizer(model.parameters(), lr=0.001)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for optimizer creation")
        import torch.optim as optim

        optimizer_class = kwargs.pop("optimizer_class", optim.AdamW)
        defaults = {
            "lr": lr,
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "weight_decay": 0.01,
        }
        if self._ipex_available and "fused" not in kwargs:
            defaults["fused"] = True
        defaults.update(kwargs)
        return optimizer_class(model_parameters, **defaults)

    def enable_jit(self, model: Any, example_inputs: Any) -> Any:
        """
        Enable JIT compilation for a model.

        Attempts scripting first, then tracing if scripting fails.

        Args:
            model: PyTorch model to compile
            example_inputs: Example inputs for tracing

        Returns:
            JIT-compiled model or original model if compilation fails

        Example:
            >>> jit_model = backend.enable_jit(model, example_inputs)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for JIT compilation")
        try:
            scripted = torch.jit.script(model)
            return scripted
        except Exception as e:
            logger.warning(f"JIT scripting failed: {e}, trying trace")
            try:
                traced = torch.jit.trace(model, example_inputs)
                return traced
            except Exception as e2:
                logger.warning(f"JIT tracing also failed: {e2}")
                return model

    def synchronize(self):
        """
        Synchronize CPU operations.

        No-op for CPU backend as operations are synchronous.

        Example:
            >>> backend.synchronize()
        """
        pass

    def cleanup(self):
        """
        Clean up backend resources.

        Runs garbage collection and resets initialization state.

        Example:
            >>> backend.cleanup()
        """
        if self._initialized:
            import gc

            gc.collect()
            self._initialized = False
            logger.info("CPU backend cleaned up")
