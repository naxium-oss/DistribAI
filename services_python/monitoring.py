"""
Monitoring and observability for DistribAI.

Provides metrics collection, health checks, and performance monitoring.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    process_count: int


@dataclass
class NodeMetrics:
    """Node-specific metrics."""

    timestamp: float
    node_id: str
    status: str
    jobs_completed: int
    jobs_failed: int
    credits_earned: float
    current_job_id: str | None
    training_steps_completed: int
    gpu_utilization: float | None
    gpu_memory_used_gb: float | None


@dataclass
class OrchestratorMetrics:
    """Orchestrator metrics."""

    timestamp: float
    connected_nodes: int
    active_jobs: int
    queued_jobs: int
    completed_jobs_24h: int
    failed_jobs_24h: int
    total_credits_distributed: float
    average_job_duration_ms: float


class MetricsCollector:
    """Collect and store metrics."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics: deque = deque(maxlen=max_history)
        self.node_metrics: dict[str, deque] = {}
        self.orchestrator_metrics: deque = deque(maxlen=max_history)
        self._collecting = False
        self._task: asyncio.Task | None = None

    async def start(self, interval_seconds: float = 60.0):
        """Start metrics collection."""
        if self._collecting:
            return

        self._collecting = True
        self._task = asyncio.create_task(self._collect_loop(interval_seconds))
        logger.info("Metrics collection started")

    async def stop(self):
        """Stop metrics collection."""
        self._collecting = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Metrics collection stopped")

    async def _collect_loop(self, interval: float):
        """Background metrics collection loop."""
        while self._collecting:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(interval)
            except (OSError, psutil.Error) as e:
                logger.error("Metrics collection error: %s", e)
                await asyncio.sleep(5)  # Shorter sleep on error

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net_io = psutil.net_io_counters()

            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_percent=mem.percent,
                memory_used_gb=mem.used / (1024**3),
                memory_total_gb=mem.total / (1024**3),
                disk_percent=disk.percent,
                network_io_sent_mb=net_io.bytes_sent / (1024**2),
                network_io_recv_mb=net_io.bytes_recv / (1024**2),
                process_count=len(psutil.pids()),
            )

            self.system_metrics.append(metrics)
            return metrics

        except (OSError, psutil.Error) as e:
            logger.error("Failed to collect system metrics: %s", e)
            raise

    def record_node_metrics(self, node_id: str, metrics: NodeMetrics):
        """Record metrics for a specific node."""
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = deque(maxlen=self.max_history)
        self.node_metrics[node_id].append(metrics)

    def record_orchestrator_metrics(self, metrics: OrchestratorMetrics):
        """Record orchestrator metrics."""
        self.orchestrator_metrics.append(metrics)

    def get_system_metrics_summary(self, minutes: int = 5) -> dict[str, Any]:
        """Get summary of system metrics for the last N minutes."""
        cutoff = time.time() - (minutes * 60)
        recent = [m for m in self.system_metrics if m.timestamp > cutoff]

        if not recent:
            return {"error": "No metrics available"}

        return {
            "period_minutes": minutes,
            "samples": len(recent),
            "cpu_percent_avg": sum(m.cpu_percent for m in recent) / len(recent),
            "cpu_percent_max": max(m.cpu_percent for m in recent),
            "memory_percent_avg": sum(m.memory_percent for m in recent) / len(recent),
            "memory_percent_max": max(m.memory_percent for m in recent),
            "disk_percent": recent[-1].disk_percent if recent else 0,
            "timestamp": datetime.now().isoformat(),
        }

    def get_node_metrics(self, node_id: str, count: int = 10) -> list[dict]:
        """Get recent metrics for a specific node."""
        if node_id not in self.node_metrics:
            return []

        recent = list(self.node_metrics[node_id])[-count:]
        return [asdict(m) for m in recent]

    def get_orchestrator_summary(self) -> dict[str, Any]:
        """Get summary of orchestrator metrics."""
        if not self.orchestrator_metrics:
            return {"error": "No orchestrator metrics available"}

        recent = list(self.orchestrator_metrics)[-10:]
        return {
            "samples": len(recent),
            "connected_nodes_avg": sum(m.connected_nodes for m in recent) / len(recent),
            "active_jobs_avg": sum(m.active_jobs for m in recent) / len(recent),
            "queued_jobs_current": recent[-1].queued_jobs if recent else 0,
            "total_credits_distributed": sum(m.total_credits_distributed for m in recent),
            "timestamp": datetime.now().isoformat(),
        }


class HealthChecker:
    """System health checking."""

    def __init__(self):
        self.checks: dict[str, callable] = {}
        self.last_results: dict[str, dict] = {}

    def register_check(self, name: str, check_func: callable):
        """Register a health check."""
        self.checks[name] = check_func
        logger.info("Registered health check: %s", name)

    async def run_all_checks(self) -> dict[str, Any]:
        """Run all registered health checks."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {},
        }

        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    healthy, message = await check_func()
                else:
                    healthy, message = check_func()

                results["checks"][name] = {
                    "status": "healthy" if healthy else "unhealthy",
                    "message": message,
                }

                if not healthy:
                    results["overall_status"] = "degraded"

            except Exception as e:
                results["checks"][name] = {"status": "error", "message": str(e)}
                results["overall_status"] = "degraded"

        self.last_results = results
        return results

    def get_last_results(self) -> dict[str, Any]:
        """Get results from last health check run."""
        return self.last_results


class PerformanceProfiler:
    """Profile performance of critical operations."""

    def __init__(self):
        self.profiles: dict[str, dict] = {}
        self.active_timers: dict[str, float] = {}

    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.active_timers[operation] = time.perf_counter()

    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.active_timers:
            return 0.0

        duration = time.perf_counter() - self.active_timers[operation]
        del self.active_timers[operation]

        # Store profile
        if operation not in self.profiles:
            self.profiles[operation] = {
                "count": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
            }

        profile = self.profiles[operation]
        profile["count"] += 1
        profile["total_time"] += duration
        profile["min_time"] = min(profile["min_time"], duration)
        profile["max_time"] = max(profile["max_time"], duration)

        return duration

    def get_profile(self, operation: str) -> dict | None:
        """Get performance profile for an operation."""
        if operation not in self.profiles:
            return None

        profile = self.profiles[operation]
        count = profile["count"]

        return {
            "operation": operation,
            "count": count,
            "avg_time_ms": (profile["total_time"] / count * 1000) if count > 0 else 0,
            "min_time_ms": profile["min_time"] * 1000,
            "max_time_ms": profile["max_time"] * 1000,
            "total_time_ms": profile["total_time"] * 1000,
        }

    def get_all_profiles(self) -> list[dict]:
        """Get all performance profiles."""
        return [self.get_profile(op) for op in self.profiles.keys()]

    def reset_profiles(self):
        """Reset all profiles."""
        self.profiles.clear()
        self.active_timers.clear()


# Global instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()
profiler = PerformanceProfiler()


def setup_default_health_checks():
    """Setup default health checks."""

    def check_disk_space():
        """Check disk space availability."""
        disk = psutil.disk_usage("/")
        healthy = disk.percent < 90
        return healthy, f"Disk usage: {disk.percent}%"

    def check_memory():
        """Check memory availability with enhanced monitoring."""
        mem = psutil.virtual_memory()
        healthy = mem.percent < 95

        # Additional memory checks
        available_gb = mem.available / (1024**3)
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)

        # Check if available memory is critically low (< 1GB)
        if available_gb < 1.0:
            healthy = False

        status_msg = f"Memory: {mem.percent:.1f}% ({used_gb:.1f}GB/{total_gb:.1f}GB, {available_gb:.1f}GB available)"
        return healthy, status_msg

    health_checker.register_check("disk_space", check_disk_space)
    health_checker.register_check("memory", check_memory)


# Setup default checks on import
setup_default_health_checks()
