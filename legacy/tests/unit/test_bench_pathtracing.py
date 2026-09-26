"""Tests for pathtracing benchmark module."""

import pytest

try:
    from worker.src.benchmark import bench_pathtracing

    HAS_BENCH_PATHTRACING = True
except ImportError:
    HAS_BENCH_PATHTRACING = False
    bench_pathtracing = None


@pytest.mark.skipif(not HAS_BENCH_PATHTRACING, reason="bench_pathtracing not available")
def test_bench_pathtracing_import():
    """Test pathtracing benchmark module imports."""
    assert bench_pathtracing is not None


@pytest.mark.skipif(not HAS_BENCH_PATHTRACING, reason="bench_pathtracing not available")
def test_sphere_geom_exists():
    """Test sphere geometry constant exists."""
    assert hasattr(bench_pathtracing, "_SPHERE_GEOM")
    assert len(bench_pathtracing._SPHERE_GEOM) > 0


@pytest.mark.skipif(not HAS_BENCH_PATHTRACING, reason="bench_pathtracing not available")
def test_sphere_albedo_exists():
    """Test sphere albedo constant exists."""
    assert hasattr(bench_pathtracing, "_SPHERE_ALBEDO")
    assert len(bench_pathtracing._SPHERE_ALBEDO) > 0


@pytest.mark.skipif(not HAS_BENCH_PATHTRACING, reason="bench_pathtracing not available")
def test_emit():
    """Test emit function exists."""
    assert hasattr(bench_pathtracing, "emit")


@pytest.mark.skipif(not HAS_BENCH_PATHTRACING, reason="bench_pathtracing not available")
def test_log_score():
    """Test log_score function."""
    result = bench_pathtracing.log_score(10.0, 1.0, 100.0)
    assert isinstance(result, float)
    assert 0 <= result <= 100


@pytest.mark.skipif(not HAS_BENCH_PATHTRACING, reason="bench_pathtracing not available")
def test_benchmark_gpu_skips_without_cuda():
    """Test GPU benchmark skips when CUDA not available."""
    if not bench_pathtracing._HAS_CUDA:
        result = bench_pathtracing.benchmark_gpu()
        assert result is None
