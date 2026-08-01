"""
Re-Benchmark Trigger System for DistribAI

Implements automatic re-benchmarking triggers to ensure node performance
characteristics stay current over time.

Triggers:
- Every 7 days (scheduled re-benchmarking)
- On driver upgrade detection
- When load balancer detects >25% throughput deviation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRecord:
    """
    Record of a completed benchmark for a node.

    Attributes:
        node_id: Node identifier
        benchmark_json: JSON string containing benchmark results
        driver_version: GPU driver version
        compute_score: Computed performance score
        timestamp: Benchmark completion timestamp

    Example:
        record = BenchmarkRecord(
            node_id="worker-001",
            benchmark_json='{"score": 100}',
            driver_version="535.104.05",
            compute_score=85.5,
            timestamp=time.time()
        )
    """

    node_id: str
    benchmark_json: str
    driver_version: str
    compute_score: float
    timestamp: float

    def to_dict(self) -> dict:
        """
        Convert benchmark record to dictionary.

        Returns:
            Dictionary with benchmark data including age in days

        Example:
            >>> data = record.to_dict()
            >>> print(f"Age: {data['age_days']} days")
        """
        return {
            "node_id": self.node_id,
            "benchmark": json.loads(self.benchmark_json) if self.benchmark_json else {},
            "driver_version": self.driver_version,
            "compute_score": self.compute_score,
            "timestamp": self.timestamp,
            "age_days": (time.time() - self.timestamp) / 86400,
        }


@dataclass
class NodePerformanceMetrics:
    """
    Tracks performance metrics for a node to detect deviations.

    Attributes:
        node_id: Node identifier
        expected_throughput: Expected tasks per hour
        actual_task_times: List of actual task completion times
        last_updated: Last metrics update timestamp

    Example:
        metrics = NodePerformanceMetrics(
            node_id="worker-001",
            expected_throughput=10.0
        )
        metrics.actual_task_times.append(300)  # 5 minutes
    """

    node_id: str
    expected_throughput: float
    actual_task_times: list[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def average_task_time(self) -> float:
        """
        Calculate average task completion time.

        Returns:
            Average task time in seconds

        Example:
            >>> avg = metrics.average_task_time()
            >>> print(f"Average task time: {avg} seconds")
        """
        if not self.actual_task_times:
            return 0.0
        return sum(self.actual_task_times) / len(self.actual_task_times)

    def throughput_deviation_percent(self) -> float:
        """
        Calculate throughput deviation from expected.

        Returns:
            Percentage deviation (positive = slower than expected)

        Example:
            >>> deviation = metrics.throughput_deviation_percent()
            >>> if deviation > 25:
            ...     print("Significant deviation detected")
        """
        if not self.actual_task_times or self.expected_throughput <= 0:
            return 0.0
        actual_throughput = 3600 / self.average_task_time() if self.average_task_time() > 0 else 0
        if actual_throughput == 0:
            return 0.0
        deviation = (self.expected_throughput - actual_throughput) / self.expected_throughput
        return deviation * 100


class RebenchmarkTriggerManager:
    """
    Manages re-benchmark triggers for worker nodes.

    Automatically triggers re-benchmarking when:
    - Benchmark is older than 7 days
    - Driver version changed (compared to last benchmark)
    - Throughput deviation exceeds 25% from expected

    Attributes:
        benchmark_history: Historical benchmarks for each node
        current_benchmarks: Most recent benchmark for each node
        performance_metrics: Performance tracking for deviation detection
        pending_rebenchmarks: Set of nodes pending re-benchmark

    Example:
        manager = RebenchmarkTriggerManager()
        manager.record_benchmark("node-001", benchmark_json, "535.104.05", 85.5)
        needs_rebenchmark = manager.check_rebenchmark_needed("node-001")
    """

    REBENCHMARK_INTERVAL_SECONDS = 604800
    THROUGHPUT_DEVIATION_THRESHOLD = 25.0
    MIN_TASKS_FOR_DEVIATION_CHECK = 5

    def __init__(self):
        """
        Initialize the re-benchmark trigger manager.

        Example:
            >>> manager = RebenchmarkTriggerManager()
        """
        self.benchmark_history: dict[str, list[BenchmarkRecord]] = {}
        self.current_benchmarks: dict[str, BenchmarkRecord] = {}
        self.performance_metrics: dict[str, NodePerformanceMetrics] = {}
        self.pending_rebenchmarks: set[str] = set()

    def record_benchmark(
        self, node_id: str, benchmark_json: str, driver_version: str, compute_score: float
    ) -> None:
        """
        Record a new benchmark for a node.

        Args:
            node_id: Node identifier
            benchmark_json: Full benchmark results JSON
            driver_version: GPU driver version
            compute_score: Computed performance score

        Example:
            >>> manager.record_benchmark(
            ...     node_id="worker-001",
            ...     benchmark_json='{"score": 100}',
            ...     driver_version="535.104.05",
            ...     compute_score=85.5
            ... )
        """
        record = BenchmarkRecord(
            node_id=node_id,
            benchmark_json=benchmark_json,
            driver_version=driver_version,
            compute_score=compute_score,
            timestamp=time.time(),
        )
        if node_id not in self.benchmark_history:
            self.benchmark_history[node_id] = []
        self.benchmark_history[node_id].append(record)
        if len(self.benchmark_history[node_id]) > 10:
            self.benchmark_history[node_id] = self.benchmark_history[node_id][-10:]
        self.current_benchmarks[node_id] = record
        if node_id not in self.performance_metrics:
            expected_tph = compute_score * 0.1
            self.performance_metrics[node_id] = NodePerformanceMetrics(
                node_id=node_id, expected_throughput=expected_tph
            )
        self.pending_rebenchmarks.discard(node_id)
        logger.info(
            f"Benchmark recorded for {node_id[:20]}... "
            f"(driver: {driver_version}, score: {compute_score:.1f})"
        )

    def record_task_completion(self, node_id: str, wall_seconds: float) -> None:
        """
        Record task completion time for throughput tracking.

        Used to detect performance deviations that may indicate
        hardware issues or driver problems requiring re-benchmarking.

        Args:
            node_id: Node that completed the task
            wall_seconds: Actual wall clock time for task completion

        Example:
            >>> manager.record_task_completion("worker-001", 300)
        """
        if node_id not in self.performance_metrics:
            return
        metrics = self.performance_metrics[node_id]
        metrics.actual_task_times.append(wall_seconds)
        metrics.last_updated = time.time()
        if len(metrics.actual_task_times) > 20:
            metrics.actual_task_times = metrics.actual_task_times[-20:]

    def check_rebenchmark_needed(
        self, node_id: str, current_driver_version: str = ""
    ) -> tuple[bool, list[str]]:
        """
        Check if a node needs to be re-benchmarked.

        Evaluates all trigger conditions:
        - Benchmark age (older than 7 days)
        - Driver version change
        - Throughput deviation (>25%)

        Args:
            node_id: Node identifier to check
            current_driver_version: Current driver version (if known)

        Returns:
            Tuple of (needs_rebenchmark, list_of_reasons)

        Example:
            >>> needs_rebenchmark, reasons = manager.check_rebenchmark_needed("worker-001")
            >>> if needs_rebenchmark:
            ...     print(f"Reasons: {reasons}")
        """
        reasons = []
        if node_id not in self.current_benchmarks:
            return True, ["No benchmark on record"]
        current = self.current_benchmarks[node_id]
        now = time.time()
        age_seconds = now - current.timestamp
        if age_seconds > self.REBENCHMARK_INTERVAL_SECONDS:
            days_old = age_seconds / 86400
            reasons.append(f"Benchmark stale ({days_old:.1f} days old)")
        if current_driver_version and current.driver_version != current_driver_version:
            reasons.append(
                f"Driver changed from {current.driver_version} to {current_driver_version}"
            )
        if node_id in self.performance_metrics:
            metrics = self.performance_metrics[node_id]
            if len(metrics.actual_task_times) >= self.MIN_TASKS_FOR_DEVIATION_CHECK:
                deviation = metrics.throughput_deviation_percent()
                if abs(deviation) > self.THROUGHPUT_DEVIATION_THRESHOLD:
                    direction = "slower" if deviation > 0 else "faster"
                    reasons.append(
                        f"Throughput deviation: {abs(deviation):.1f}% {direction} than expected"
                    )
        return len(reasons) > 0, reasons

    def schedule_rebenchmark(self, node_id: str) -> bool:
        """
        Schedule a node for re-benchmarking.

        Adds the node to the pending re-benchmark set for later processing.

        Args:
            node_id: Node identifier to schedule

        Returns:
            True if node was scheduled

        Example:
            >>> manager.schedule_rebenchmark("worker-001")
        """
        self.pending_rebenchmarks.add(node_id)
        logger.info("Re-benchmark scheduled for %s...", node_id[:20])
        return True

    def is_pending_rebenchmark(self, node_id: str) -> bool:
        """
        Check if a node is pending re-benchmark.

        Args:
            node_id: Node identifier to check

        Returns:
            True if node is in pending re-benchmark set

        Example:
            >>> if manager.is_pending_rebenchmark("worker-001"):
            ...     print("Re-benchmark pending")
        """
        return node_id in self.pending_rebenchmarks

    def complete_rebenchmark(self, node_id: str) -> None:
        """
        Mark a re-benchmark as completed.

        Removes the node from the pending re-benchmark set.

        Args:
            node_id: Node identifier to mark as completed

        Example:
            >>> manager.complete_rebenchmark("worker-001")
        """
        self.pending_rebenchmarks.discard(node_id)

    def get_benchmark_status(self, node_id: str) -> dict:
        """
        Get benchmark status for a node.

        Args:
            node_id: Node identifier to query

        Returns:
            Dictionary with benchmark status information

        Example:
            >>> status = manager.get_benchmark_status("worker-001")
            >>> print(f"Age: {status.get('age_days', 0):.1f} days")
        """
        if node_id not in self.current_benchmarks:
            return {
                "node_id": node_id,
                "has_benchmark": False,
                "needs_rebenchmark": True,
                "reasons": ["No benchmark on record"],
            }
        current = self.current_benchmarks[node_id]
        needs_rebench, reasons = self.check_rebenchmark_needed(node_id)
        perf_data = None
        if node_id in self.performance_metrics:
            metrics = self.performance_metrics[node_id]
            perf_data = {
                "expected_throughput_tph": round(metrics.expected_throughput, 2),
                "average_task_time_sec": round(metrics.average_task_time(), 2),
                "tasks_tracked": len(metrics.actual_task_times),
                "deviation_percent": round(metrics.throughput_deviation_percent(), 2),
            }
        return {
            "node_id": node_id,
            "has_benchmark": True,
            "current_benchmark": current.to_dict(),
            "next_benchmark_due": current.timestamp + self.REBENCHMARK_INTERVAL_SECONDS,
            "needs_rebenchmark": needs_rebench,
            "reasons": reasons,
            "is_pending": self.is_pending_rebenchmark(node_id),
            "performance": perf_data,
        }

    def get_all_pending(self) -> list[str]:
        """
        Get all nodes pending re-benchmark.

        Returns:
            List of node IDs pending re-benchmark

        Example:
            >>> pending = manager.get_all_pending()
            >>> print(f"Pending re-benchmarks: {len(pending)}")
        """
        return list(self.pending_rebenchmarks)

    def get_stats(self) -> dict:
        """
        Get statistics about re-benchmark triggers.

        Returns:
            Dictionary with statistics including total nodes, pending count,
            and nodes needing re-benchmark

        Example:
            >>> stats = manager.get_stats()
            >>> print(f"Total nodes: {stats['total_nodes']}")
            >>> print(f"Pending: {stats['pending']}")
        """
        total_nodes = len(self.current_benchmarks)
        pending = len(self.pending_rebenchmarks)
        needs_rebench_count = 0
        for node_id in self.current_benchmarks:
            needs, _ = self.check_rebenchmark_needed(node_id)
            if needs:
                needs_rebench_count += 1
        if self.current_benchmarks:
            ages = [time.time() - b.timestamp for b in self.current_benchmarks.values()]
            avg_age_days = (sum(ages) / len(ages)) / 86400
        else:
            avg_age_days = 0
        return {
            "total_nodes_with_benchmarks": total_nodes,
            "pending_rebenchmarks": pending,
            "nodes_needing_rebenchmark": needs_rebench_count,
            "average_benchmark_age_days": round(avg_age_days, 1),
            "rebenchmark_interval_days": self.REBENCHMARK_INTERVAL_SECONDS / 86400,
            "throughput_deviation_threshold_percent": self.THROUGHPUT_DEVIATION_THRESHOLD,
        }
