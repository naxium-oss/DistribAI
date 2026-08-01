from __future__ import annotations

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from worker.src.daemon.byzantine_detector import (
        AggregationMethod,
        AnomalyScore,
        CoordinateWiseMedian,
        Krum,
        MultiKrum,
        RobustAggregator,
        TrimmedMean,
    )

    HAS_BYZANTINE = True
except ImportError:
    HAS_BYZANTINE = False
    # Define dummy classes for test collection
    AggregationMethod = None
    CoordinateWiseMedian = None
    TrimmedMean = None
    Krum = None
    MultiKrum = None
    RobustAggregator = None
    AnomalyScore = None


def _three_updates():
    if not HAS_TORCH:
        return {}
    g = torch.ones(8)
    return {
        "a": g * 1.0,
        "b": g * 1.05,
        "c": g * 0.95,
    }


@pytest.mark.skipif(not HAS_BYZANTINE, reason="byzantine_detector not available")
def test_imports():
    assert AggregationMethod.KRUM is not None
    assert CoordinateWiseMedian is not None


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_BYZANTINE, reason="torch or byzantine_detector not available"
)
def test_coordinate_median_aggregate():
    det = CoordinateWiseMedian()
    out = det.aggregate(_three_updates())
    assert out.shape == (8,)
    assert torch.allclose(out, torch.ones(8), atol=0.2)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_BYZANTINE, reason="torch or byzantine_detector not available"
)
def test_coordinate_median_empty():
    det = CoordinateWiseMedian()
    out = det.aggregate({})
    assert out.numel() == 0


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_BYZANTINE, reason="torch or byzantine_detector not available"
)
def test_trimmed_mean_aggregate():
    det = TrimmedMean(max_byzantine_fraction=0.2)
    updates = {f"n{i}": torch.ones(10) * float(i) for i in range(5)}
    out = det.aggregate(updates)
    assert out.shape == (10,)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_BYZANTINE, reason="torch or byzantine_detector not available"
)
def test_trimmed_mean_detect_anomalies():
    det = TrimmedMean(max_byzantine_fraction=0.2)
    scores = det.detect_anomalies(_three_updates())
    assert len(scores) == 3
    assert all(isinstance(s, AnomalyScore) for s in scores)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_BYZANTINE, reason="torch or byzantine_detector not available"
)
def test_krum_aggregate():
    det = Krum(max_byzantine_fraction=0.2)
    out = det.aggregate(_three_updates())
    assert out.shape == (8,)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_BYZANTINE, reason="torch or byzantine_detector not available"
)
def test_multi_krum_aggregate():
    det = MultiKrum(max_byzantine_fraction=0.2, num_selected=2)
    updates = {f"n{i}": torch.randn(4) for i in range(6)}
    out = det.aggregate(updates)
    assert out.shape == (4,)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_BYZANTINE, reason="torch or byzantine_detector not available"
)
def test_robust_aggregator_trimmed_mean():
    agg = RobustAggregator(method=AggregationMethod.TRIMMED_MEAN)
    out = agg.aggregate(_three_updates())
    assert out.shape == (8,)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_BYZANTINE, reason="torch or byzantine_detector not available"
)
def test_robust_aggregator_coordinate_median():
    agg = RobustAggregator(method=AggregationMethod.COORD_MEDIAN)
    out = agg.aggregate(_three_updates())
    assert out.shape == (8,)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_BYZANTINE, reason="torch or byzantine_detector not available"
)
def test_detect_anomalies_krum():
    agg = RobustAggregator(method=AggregationMethod.KRUM)
    scores = agg.detect_anomalies(_three_updates())
    assert len(scores) == 3
