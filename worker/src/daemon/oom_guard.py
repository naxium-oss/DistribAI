"""
Out of Memory (OOM) Guard for DistribAI Worker

Provides centralized OOM detection, cleanup, and recovery strategies
across different compute backends (CUDA, MPS, CPU).

Design notes
------------
A ``with`` block cannot be re-executed by its context manager, so
:meth:`OOMGuard.guard` performs exactly one attempt: it classifies the
failure, records telemetry, runs the strategy's memory cleanup, and
re-raises the original error for the caller to act on (the previous
implementation tried to retry by yielding twice, which ``contextlib``
rejects with ``generator didn't stop after throw()`` — the recovery path
was unreachable). Callers that want genuine retries wrap the retryable
unit of work in :meth:`OOMGuard.run`, which re-invokes the callable with
cleanup between attempts, optionally shrinking the workload via an
``on_retry`` hook (e.g. halving the batch size).
"""

import gc
import logging
import platform
import time
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Any

import psutil
import torch

logger = logging.getLogger(__name__)


def is_oom_error(error: BaseException) -> bool:
    """Classify an exception as an out-of-memory failure on any backend.

    Covers ``torch.cuda.OutOfMemoryError``, MPS/CPU ``RuntimeError`` texts
    that mention running out of memory, and the built-in ``MemoryError``.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(error, MemoryError):
        return True
    if isinstance(error, RuntimeError):
        return "out of memory" in str(error).lower()
    return False


def _oom_backend_name(error: BaseException) -> str:
    """Best-effort backend label for telemetry and logs."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "CUDA"
    if isinstance(error, MemoryError):
        return "CPU"
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
        return "MPS"
    return "CPU"


class OOMStrategy(Enum):
    """OOM recovery strategies (they select the cleanup aggressiveness)."""

    REDUCE_BATCH_SIZE = "reduce_batch_size"
    FALLBACK_TO_CPU = "fallback_to_cpu"
    CHECKPOINT_AND_RETRY = "checkpoint_and_retry"
    GRACEFUL_FAILURE = "graceful_failure"


class OOMGuard:
    """
    Centralized OOM guard with configurable recovery strategies.

    Detects and handles OOM errors across CUDA, MPS, and CPU backends
    with cleanup strategies, retry orchestration, and detailed telemetry.

    Attributes:
        strategy: Primary recovery strategy to use
        max_retries: Maximum number of recovery attempts for :meth:`run`
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
        Single-attempt context manager: telemetry + cleanup on OOM.

        On an OOM failure the guard records telemetry, runs the configured
        cleanup strategy (freeing caches so the *caller's* retry has room),
        and re-raises the original exception. Use :meth:`run` when the
        operation itself should be retried automatically.

        Args:
            operation_name: Name of the operation for telemetry

        Raises:
            The original OOM (or any other) exception from the wrapped code.
        """
        if self.enable_telemetry:
            self._capture_memory_snapshot(f"pre_{operation_name}")
        try:
            yield
        except BaseException as error:  # noqa: BLE001 - classified below, always re-raised
            if is_oom_error(error):
                self.handle_oom(error, operation_name)
            raise

    def run(
        self,
        fn: Callable[..., Any],
        *args: Any,
        operation_name: str = "operation",
        on_retry: Callable[[int], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Invoke ``fn`` with genuine OOM retries and cleanup between attempts.

        Args:
            fn: The unit of work; re-invoked after each OOM cleanup.
            operation_name: Name used in telemetry and logs.
            on_retry: Optional hook called with the upcoming attempt number
                (1-based) before each retry — the natural place to shrink
                batch sizes or drop caches owned by the caller.

        Returns:
            Whatever ``fn`` returns.

        Raises:
            RuntimeError: When every retry attempt also hit OOM.
            Exception: Non-OOM errors propagate immediately.
        """
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            if attempt and on_retry is not None:
                on_retry(attempt)
            try:
                if self.enable_telemetry and attempt:
                    self._capture_memory_snapshot(f"retry{attempt}_{operation_name}")
                return fn(*args, **kwargs)
            except BaseException as error:  # noqa: BLE001 - non-OOM re-raised below
                if not is_oom_error(error):
                    raise
                last_error = error
                self.handle_oom(error, operation_name)
        raise RuntimeError(
            f"OOM recovery failed after {self.max_retries} attempts for {operation_name}"
        ) from last_error

    def handle_oom(self, error: BaseException, operation_name: str) -> bool:
        """Record one OOM event and run the strategy's memory cleanup.

        Returns:
            True when cleanup completed without secondary errors.
        """
        backend = _oom_backend_name(error)
        if self.enable_telemetry:
            self.telemetry["oom_count"] += 1
            self.telemetry["last_oom_time"] = time.time()
            self._capture_memory_snapshot(f"oom_{backend}_{operation_name}")

        logger.warning("OOM error in %s during %s: %s", backend, operation_name, error)

        if self.strategy == OOMStrategy.REDUCE_BATCH_SIZE:
            recovery_success = self._recover_reduce_batch_size()
        elif self.strategy == OOMStrategy.FALLBACK_TO_CPU:
            recovery_success = self._recover_fallback_to_cpu()
        elif self.strategy == OOMStrategy.CHECKPOINT_AND_RETRY:
            recovery_success = self._recover_checkpoint_and_retry()
        else:
            recovery_success = self._recover_graceful_failure()

        if recovery_success:
            self.telemetry["recovery_success"] += 1
            logger.info("OOM cleanup successful for %s", operation_name)
        else:
            self.telemetry["recovery_failures"] += 1
            logger.error("OOM cleanup failed for %s", operation_name)
        return recovery_success

    def _recover_reduce_batch_size(self) -> bool:
        """Free device caches so the caller can retry with a smaller batch."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            gc.collect()
            return True
        except Exception as e:
            logger.error(f"Failed to reduce batch size recovery: {e}")
            return False

    def _recover_fallback_to_cpu(self) -> bool:
        """Recover by moving live accelerator tensors to CPU."""
        try:
            if torch.cuda.is_available():
                for obj in gc.get_objects():
                    if isinstance(obj, torch.Tensor) and obj.is_cuda:
                        obj.data = obj.data.cpu()

            if hasattr(torch.mps, "is_available") and torch.mps.is_available():
                for obj in gc.get_objects():
                    if isinstance(obj, torch.Tensor) and obj.device.type == "mps":
                        obj.data = obj.data.cpu()

            gc.collect()
            return True
        except Exception as e:
            logger.error(f"Failed CPU fallback recovery: {e}")
            return False

    def _recover_checkpoint_and_retry(self) -> bool:
        """Aggressive cleanup ahead of a checkpoint reload by the caller."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            # Multiple GC passes to break reference cycles promptly.
            for _ in range(3):
                gc.collect()

            return True
        except Exception as e:
            logger.error(f"Failed checkpoint recovery: {e}")
            return False

    def _recover_graceful_failure(self) -> bool:
        """Best-effort cleanup with no expectation of retrying."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            gc.collect()
            return True
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
            except (RuntimeError, AttributeError) as exc:
                logger.debug("CUDA memory snapshot unavailable: %s", exc)

        if hasattr(torch.mps, "is_available") and torch.mps.is_available():
            snapshot["mps_available"] = True

        try:
            memory = psutil.virtual_memory()
            snapshot.update(
                {
                    "system_total_mb": memory.total // (1024 * 1024),
                    "system_available_mb": memory.available // (1024 * 1024),
                    "system_percent": memory.percent,
                }
            )
        except (OSError, psutil.Error) as exc:
            logger.debug("System memory snapshot unavailable: %s", exc)

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
    Decorator applying real OOM retries (via :meth:`OOMGuard.run`) to a function.

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
            return guard.run(func, *args, operation_name=operation_name, **kwargs)

        return wrapper

    return decorator
