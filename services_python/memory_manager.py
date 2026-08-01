"""
Memory Management Utilities for DistribAI

Provides comprehensive memory monitoring, OOM handling, and cleanup utilities
to ensure stable operation across different hardware configurations.
"""

import asyncio
import gc
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import psutil

from .platform_utils import (
    get_platform_specific_limits,
    get_platform_specific_optimizations,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


class MemoryManager:
    """Comprehensive memory management for the DistribAI system."""

    def __init__(self, memory_threshold_gb: float = None, cleanup_interval: int = None):
        """
        Initialize memory manager with platform-specific settings.

        Args:
            memory_threshold_gb: Memory usage threshold in GB for automatic cleanup (auto-determined if None)
            cleanup_interval: Interval in seconds for periodic cleanup (auto-determined if None)
        """
        # Get platform-specific settings
        platform_limits = get_platform_specific_limits()
        platform_opts = get_platform_specific_optimizations()

        self.memory_threshold_gb = memory_threshold_gb or (platform_limits["max_memory_gb"] * 0.8)
        self.cleanup_interval = cleanup_interval or (
            300 if platform_opts["use_multiprocessing"] else 600
        )
        self._cleanup_task: asyncio.Task | None = None
        self._memory_history: list[dict[str, Any]] = []
        self._max_history_size = 1000

    async def start_background_cleanup(self):
        """Start background memory cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())
            logger.info("Started background memory cleanup task")

    async def stop_background_cleanup(self):
        """Stop background memory cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped background memory cleanup task")

    async def _background_cleanup_loop(self):
        """Background task that performs periodic memory cleanup."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.check_and_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background cleanup: {e}")

    def get_memory_info(self) -> dict[str, Any]:
        """Get comprehensive memory usage information."""
        mem = psutil.virtual_memory()
        result = {
            "timestamp": datetime.now(UTC).isoformat(),
            "system_memory": {
                "percent": mem.percent,
                "used_gb": mem.used / (1024**3),
                "available_gb": mem.available / (1024**3),
                "total_gb": mem.total / (1024**3),
            },
            "process_memory": {
                "rss_mb": psutil.Process().memory_info().rss / (1024**2),
                "vms_mb": psutil.Process().memory_info().vms / (1024**2),
            },
        }

        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                allocated = torch.cuda.memory_allocated()
                result["gpu_memory"] = {
                    "allocated_gb": allocated / (1024**3),
                    "cached_gb": torch.cuda.memory_reserved() / (1024**3),
                    "total_gb": total_mem / (1024**3),
                    "utilization_percent": (allocated / total_mem) * 100,
                }
            except Exception as e:
                logger.warning(
                    "CUDA reported available but memory stats failed (%s); skipping gpu_memory.",
                    e,
                )

        return result

    def _record_memory_state(self, memory_info: dict[str, Any]):
        """Record memory state for historical tracking."""
        self._memory_history.append(memory_info)
        if len(self._memory_history) > self._max_history_size:
            self._memory_history.pop(0)

    async def check_and_cleanup(self) -> bool:
        """
        Check memory usage and perform cleanup if needed.

        Returns:
            True if cleanup was performed, False otherwise
        """
        memory_info = self.get_memory_info()
        self._record_memory_state(memory_info)

        cleanup_performed = False

        # Check system memory
        if memory_info["system_memory"]["available_gb"] < self.memory_threshold_gb:
            logger.warning(
                f"Low system memory: {memory_info['system_memory']['available_gb']:.2f}GB available"
            )
            await self.perform_cleanup("system_memory_low")
            cleanup_performed = True

        # Check GPU memory if available
        if "gpu_memory" in memory_info:
            gpu_available = (
                memory_info["gpu_memory"]["total_gb"] - memory_info["gpu_memory"]["allocated_gb"]
            )
            if gpu_available < self.memory_threshold_gb:
                logger.warning(f"Low GPU memory: {gpu_available:.2f}GB available")
                await self.perform_cleanup("gpu_memory_low")
                cleanup_performed = True

        return cleanup_performed

    async def perform_cleanup(self, reason: str = "manual"):
        """
        Perform comprehensive memory cleanup.

        Args:
            reason: Reason for cleanup (for logging)
        """
        logger.info(f"Performing memory cleanup: {reason}")

        # Force Python garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collected {collected} objects")

        # Clear PyTorch caches if available
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                try:
                    before_memory = torch.cuda.memory_allocated()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    after_memory = torch.cuda.memory_allocated()
                    freed_mb = (before_memory - after_memory) / (1024**2)
                    logger.debug(f"Freed {freed_mb:.2f}MB from CUDA cache")
                except Exception as e:
                    logger.warning(
                        "CUDA cleanup skipped (%s); likely CPU-only build or unavailable runtime.",
                        e,
                    )

            # Clear MPS cache if available
            if (
                hasattr(torch, "mps")
                and hasattr(torch.mps, "is_available")
                and torch.mps.is_available()
            ):
                torch.mps.empty_cache()
                logger.debug("Cleared MPS cache")

        # Additional cleanup for large objects
        await self._cleanup_large_objects()

        logger.info("Memory cleanup completed")

    async def _cleanup_large_objects(self):
        """Hook for subclasses to drop tracked tensors or buffers."""
        # Base MemoryManager owns no pooled objects; subclasses override when needed.
        return None

    async def handle_oom_with_retry(
        self, operation: Callable, operation_name: str, max_retries: int = 2
    ) -> Any:
        """
        Execute an operation with OOM handling and retry logic.

        Args:
            operation: The operation to execute
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retries after OOM

        Returns:
            Result of the operation

        Raises:
            The original exception if all retries fail
        """
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                else:
                    return operation()
            except Exception as e:
                if self._is_oom_error(e):
                    logger.warning(f"OOM in {operation_name} (attempt {attempt + 1}): {e}")

                    if attempt < max_retries:
                        await self.perform_cleanup(f"oom_retry_{operation_name}")
                        # Add small delay to allow memory to be freed
                        await asyncio.sleep(0.1 * (attempt + 1))
                        continue
                    else:
                        logger.error(
                            f"OOM retry failed in {operation_name} after {max_retries} attempts"
                        )
                        raise
                else:
                    # Non-OOM error, re-raise immediately
                    raise

    def _is_oom_error(self, error: Exception) -> bool:
        """Check if an error is related to out of memory."""
        if TORCH_AVAILABLE:
            if isinstance(error, torch.cuda.OutOfMemoryError):
                return True
            if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
                return True

        if isinstance(error, MemoryError):
            return True

        return False

    def get_memory_trend(self, minutes: int = 60) -> dict[str, Any]:
        """
        Get memory usage trend over time.

        Args:
            minutes: Number of minutes to look back

        Returns:
            Dictionary with trend information
        """
        cutoff_time = datetime.now(UTC) - timedelta(minutes=minutes)
        recent_data = [
            entry
            for entry in self._memory_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]

        if not recent_data:
            return {"trend": "no_data", "data_points": 0}

        system_mem_values = [entry["system_memory"]["percent"] for entry in recent_data]
        avg_memory = sum(system_mem_values) / len(system_mem_values)
        max_memory = max(system_mem_values)
        min_memory = min(system_mem_values)

        # Determine trend
        if len(system_mem_values) >= 2:
            recent_avg = sum(system_mem_values[-5:]) / min(5, len(system_mem_values))
            earlier_avg = sum(system_mem_values[:5]) / min(5, len(system_mem_values))

            if recent_avg > earlier_avg + 5:
                trend = "increasing"
            elif recent_avg < earlier_avg - 5:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "trend": trend,
            "data_points": len(recent_data),
            "average_percent": avg_memory,
            "max_percent": max_memory,
            "min_percent": min_memory,
            "time_range_minutes": minutes,
        }

    def get_memory_health_score(self) -> float:
        """
        Get a health score (0-100) based on current memory usage.

        Returns:
            Health score where 100 is optimal and 0 is critical
        """
        memory_info = self.get_memory_info()

        # System memory health (70% weight)
        system_percent = memory_info["system_memory"]["percent"]
        if system_percent < 50:
            system_score = 100
        elif system_percent < 80:
            system_score = 100 - (system_percent - 50) * 1.5
        elif system_percent < 95:
            system_score = 55 - (system_percent - 80) * 2
        else:
            system_score = max(0, 25 - (system_percent - 95) * 5)

        # GPU memory health if available (30% weight)
        if "gpu_memory" in memory_info:
            gpu_percent = memory_info["gpu_memory"]["utilization_percent"]
            if gpu_percent < 60:
                gpu_score = 100
            elif gpu_percent < 85:
                gpu_score = 100 - (gpu_percent - 60) * 2
            elif gpu_percent < 95:
                gpu_score = 70 - (gpu_percent - 85) * 4
            else:
                gpu_score = max(0, 30 - (gpu_percent - 95) * 6)

            overall_score = (system_score * 0.7) + (gpu_score * 0.3)
        else:
            overall_score = system_score

        return round(overall_score, 1)


# Global memory manager instance
_memory_manager = MemoryManager()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    return _memory_manager


async def handle_oom_with_retry(
    operation: Callable, operation_name: str, max_retries: int = 2
) -> Any:
    """
    Convenience function to handle OOM with retry using global memory manager.

    Args:
        operation: The operation to execute
        operation_name: Name of the operation for logging
        max_retries: Maximum number of retries after OOM

    Returns:
        Result of the operation
    """
    return await _memory_manager.handle_oom_with_retry(operation, operation_name, max_retries)


def get_memory_info() -> dict[str, Any]:
    """Convenience function to get memory info using global memory manager."""
    return _memory_manager.get_memory_info()


async def check_and_cleanup_memory() -> bool:
    """Convenience function to check and cleanup memory using global memory manager."""
    return await _memory_manager.check_and_cleanup()


def get_memory_health_score() -> float:
    """Convenience function to get memory health score using global memory manager."""
    return _memory_manager.get_memory_health_score()
