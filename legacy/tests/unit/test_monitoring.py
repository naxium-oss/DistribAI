"""Tests for monitoring module."""

import pytest


def test_health_checker_creation():
    """Test HealthChecker creation."""
    try:
        from services_python.monitoring import HealthChecker
    except ImportError:
        pytest.skip("HealthChecker not available")
        return

    checker = HealthChecker()
    assert checker is not None


@pytest.mark.asyncio
async def test_health_checker_run_all_checks():
    """Test HealthChecker run_all_checks."""
    try:
        from services_python.monitoring import HealthChecker
    except ImportError:
        pytest.skip("HealthChecker not available")
        return

    checker = HealthChecker()
    results = await checker.run_all_checks()
    assert isinstance(results, dict)


def test_metrics_collector_creation():
    """Test MetricsCollector creation."""
    try:
        from services_python.monitoring import MetricsCollector
    except ImportError:
        pytest.skip("MetricsCollector not available")
        return

    collector = MetricsCollector()
    assert collector is not None


def test_metrics_collector_system_metrics():
    """Test MetricsCollector system metrics."""
    try:
        from services_python.monitoring import MetricsCollector
    except ImportError:
        pytest.skip("MetricsCollector not available")
        return

    collector = MetricsCollector()
    metrics = collector.get_system_metrics_summary()
    assert isinstance(metrics, dict)


def test_profiler_creation():
    """Test PerformanceProfiler creation."""
    try:
        from services_python.monitoring import PerformanceProfiler
    except ImportError:
        pytest.skip("PerformanceProfiler not available")
        return

    profiler = PerformanceProfiler()
    assert profiler is not None


def test_profiler_timer():
    """Test PerformanceProfiler timer functions."""
    try:
        from services_python.monitoring import PerformanceProfiler
    except ImportError:
        pytest.skip("PerformanceProfiler not available")
        return

    profiler = PerformanceProfiler()

    # Start and end timer
    profiler.start_timer("test_operation")
    duration = profiler.end_timer("test_operation")

    # Check duration is non-negative
    assert duration >= 0.0

    # Check that profile was recorded
    profile = profiler.get_profile("test_operation")
    assert profile is not None
    assert "count" in profile
