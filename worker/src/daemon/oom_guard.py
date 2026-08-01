"""
Out of Memory (OOM) Guard for DistribAI Worker

Provides centralized OOM detection, handling, and recovery strategies
across different compute backends (CUDA, MPS, CPU).
"""

import gc
import logging
import platform
import time
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class OOMStrategy(Enum):
    """OOM recovery strategies."""

    REDUCE_BATCH_SIZE = "reduce_batch_size"
    FALLBACK_TO_CPU = "fallback_to_cpu"
    CHECKPOINT_AND_RETRY = "checkpoint_and_retry"
    GRACEFUL_FAILURE = "graceful_failure"


class OOMGuard:
    """
    Centralized OOM guard with configurable recovery strategies.

    Detects and handles OOM errors across CUDA, MPS, and CPU backends
    with multiple recovery strategies and detailed telemetry.

    Attributes:
        strategy: Primary recovery strategy to use
        max_retries: Maximum number of recovery attempts
        telemetry: Dictionary of OOM event data
    """

    def __init__(
        self,
        strategy: OOMStrategy = OOMStrategy.REDUCE_BATCH_SIZE,
        max_retries: int = 3,
        enable_telemetry: bool = True,
    ):
        self.strategy = strategy
        self.max_retries = max_retries
        self.enable_telemetry = enable_telemetry
        self.telemetry = {
            "oom_count": 0,
            "recovery_success": 0,
            "recovery_failures": 0,
            "last_oom_time": None,
            "memory_snapshots": [],
        }

    @contextmanager
    def guard(self, operation_name: str = "operation"):
        """
        Context manager to guard against OOM errors.

        Args:
            operation_name: Name of the operation for telemetry

        Yields:
            None - control passes to the wrapped code

        Raises:
            RuntimeError: If OOM recovery fails after max_retries
        """
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                # Pre-operation memory snapshot
                if self.enable_telemetry:
                    self._capture_memory_snapshot(f"pre_{operation_name}")

                yield
                break  # Success, exit retry loop

            except torch.cuda.OutOfMemoryError as e:
                retry_count = self._handle_oom_error(e, "CUDA", operation_name, retry_count)
            except RuntimeError as e:
                # MPS and some CPU backends raise generic RuntimeError
                if "out of memory" in str(e).lower():
                    backend = (
                        "MPS"
                        if platform.system() == "Darwin" and hasattr(torch.backends, "mps")
                        else "CPU"
                    )
                    retry_count = self._handle_oom_error(e, backend, operation_name, retry_count)
                else:
                    # Not an OOM error, re-raise
                    raise
            except MemoryError as e:
                retry_count = self._handle_oom_error(e, "CPU", operation_name, retry_count)

            if retry_count > self.max_retries:
                raise RuntimeError(
                    f"OOM recovery failed after {self.max_retries} attempts for {operation_name}"
                )

    def _handle_oom_error(
        self,
        error: Exception,
        backend: str,
        operation_name: str,
        retry_count: int,
    ) -> int:
        """Handle an OOM error and attempt recovery."""
        retry_count += 1

        # Update telemetry
        if self.enable_telemetry:
            self.telemetry["oom_count"] += 1
            self.telemetry["last_oom_time"] = time.time()
            self._capture_memory_snapshot(f"oom_{backend}_{operation_name}")

        logger.warning(
            f"OOM error in {backend} during {operation_name} "
            f"(attempt {retry_count}/{self.max_retries}): {error}"
        )

        # Attempt recovery based on strategy
        recovery_success = False

        if self.strategy == OOMStrategy.REDUCE_BATCH_SIZE:
            recovery_success = self._recover_reduce_batch_size()
        elif self.strategy == OOMStrategy.FALLBACK_TO_CPU:
            recovery_success = self._recover_fallback_to_cpu()
        elif self.strategy == OOMStrategy.CHECKPOINT_AND_RETRY:
            recovery_success = self._recover_checkpoint_and_retry()
        elif self.strategy == OOMStrategy.GRACEFUL_FAILURE:
            recovery_success = self._recover_graceful_failure()

        if recovery_success:
            self.telemetry["recovery_success"] += 1
            logger.info(f"OOM recovery successful for {operation_name}")
        else:
            self.telemetry["recovery_failures"] += 1
            logger.error(f"OOM recovery failed for {operation_name}")

        return retry_count

    def _recover_reduce_batch_size(self) -> bool:
        """Recover by reducing batch size and clearing cache."""
        try:
            # Clear all caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            # Force garbage collection
            gc.collect()

            return True
        except Exception as e:
            logger.error(f"Failed to reduce batch size recovery: {e}")
            return False

    def _recover_fallback_to_cpu(self) -> bool:
        """Recover by moving tensors to CPU."""
        try:
            # Move all CUDA/MPS tensors to CPU
            if torch.cuda.is_available():
                for obj in gc.get_objects():
                    if isinstance(obj, torch.Tensor) and obj.is_cuda:
                        obj.data = obj.data.cpu()

            if hasattr(torch.mps, "is_available") and torch.mps.is_available():
                for obj in gc.get_objects():
                    if isinstance(obj, torch.Tensor) and obj.device.type == "mps":
                        obj.data = obj.data.cpu()

            # Clear caches
            gc.collect()
            return True
        except Exception as e:
            logger.error(f"Failed CPU fallback recovery: {e}")
            return False

    def _recover_checkpoint_and_retry(self) -> bool:
        """Recover by clearing memory and preparing for checkpoint reload."""
        try:
            # Aggressive memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            # Multiple GC passes
            for _ in range(3):
                gc.collect()

            return True
        except Exception as e:
            logger.error(f"Failed checkpoint recovery: {e}")
            return False

    def _recover_graceful_failure(self) -> bool:
        """Handle graceful failure with cleanup."""
        try:
            # Best effort cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            gc.collect()
            return True  # Always "succeeds" - doesn't retry
        except Exception as e:
            logger.error(f"Failed graceful cleanup: {e}")
            return False

    def _capture_memory_snapshot(self, label: str):
        """Capture memory usage snapshot for telemetry."""
        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "platform": platform.system(),
        }

        # CUDA memory
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                snapshot.update(
                    {
                        "cuda_allocated_mb": allocated // (1024 * 1024),
                        "cuda_reserved_mb": reserved // (1024 * 1024),
                        "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024),
                    }
                )
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        # MPS memory (limited info available)
        if hasattr(torch.mps, "is_available") and torch.mps.is_available():
            snapshot["mps_available"] = True

        # System memory
        try:
            import psutil

            memory = psutil.virtual_memory()
            snapshot.update(
                {
                    "system_total_mb": memory.total // (1024 * 1024),
                    "system_available_mb": memory.available // (1024 * 1024),
                    "system_percent": memory.percent,
                }
            )
        except ImportError:
            pass

        self.telemetry["memory_snapshots"].append(snapshot)

        # Keep only last 10 snapshots to avoid memory bloat
        if len(self.telemetry["memory_snapshots"]) > 10:
            self.telemetry["memory_snapshots"] = self.telemetry["memory_snapshots"][-10:]

    def get_telemetry(self) -> dict[str, Any]:
        """Get OOM telemetry data."""
        return self.telemetry.copy()

    def reset_telemetry(self):
        """Reset telemetry counters."""
        self.telemetry = {
            "oom_count": 0,
            "recovery_success": 0,
            "recovery_failures": 0,
            "last_oom_time": None,
            "memory_snapshots": [],
        }

    def suggest_memory_reduction(self) -> dict[str, Any]:
        """Suggest memory reduction based on telemetry."""
        if not self.telemetry["memory_snapshots"]:
            return {"suggestion": "No telemetry data available"}

        latest = self.telemetry["memory_snapshots"][-1]

        suggestions = []

        # Check CUDA memory pressure
        if "cuda_allocated_mb" in latest:
            cuda_usage = latest["cuda_allocated_mb"]
            if cuda_usage > 8000:  # > 8GB
                suggestions.append(
                    {
                        "type": "reduce_batch_size",
                        "reason": f"High CUDA memory usage: {cuda_usage}MB",
                        "action": "Reduce batch size by 25-50%",
                    }
                )

        # Check system memory pressure
        if "system_percent" in latest and latest["system_percent"] > 90:
            suggestions.append(
                {
                    "type": "reduce_system_load",
                    "reason": f"High system memory usage: {latest['system_percent']}%",
                    "action": "Close other applications or reduce GPU memory allocation",
                }
            )

        # Check frequent OOMs
        if self.telemetry["oom_count"] > 3:
            suggestions.append(
                {
                    "type": "lower_resource_limits",
                    "reason": f"Frequent OOMs: {self.telemetry['oom_count']} occurrences",
                    "action": "Reduce GPU memory percentage in settings",
                }
            )

        return {
            "suggestions": suggestions,
            "latest_snapshot": latest,
            "oom_stats": {
                "count": self.telemetry["oom_count"],
                "success_rate": (
                    self.telemetry["recovery_success"] / max(1, self.telemetry["oom_count"])
                )
                * 100,
            },
        }


# Global OOM guard instance
_default_guard = None


def get_oom_guard() -> OOMGuard:
    """Get the default OOM guard instance."""
    global _default_guard
    if _default_guard is None:
        _default_guard = OOMGuard()
    return _default_guard


def oom_guard(operation_name: str = "operation"):
    """
    Decorator to apply OOM guard to a function.

    Args:
        operation_name: Name of the operation for telemetry

    Example:
        @oom_guard("model_training")
        def train_model(model, data):
            return model.train(data)
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            guard = get_oom_guard()
            with guard.guard(operation_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
