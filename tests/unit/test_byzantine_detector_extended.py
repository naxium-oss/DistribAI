"""Extended tests for byzantine detector module."""

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
    AggregationMethod = None
    CoordinateWiseMedian = None
    TrimmedMean = None
    Krum = None
    MultiKrum = None
    RobustAggregator = None
    AnomalyScore = None


@pytest.mark.skipif(not HAS_BYZANTINE, reason="byzantine_detector not available")
def test_aggregation_method_enum():
    """Test AggregationMethod enum values."""
    assert AggregationMethod.KRUM is not None
    assert AggregationMethod.TRIMMED_MEAN is not None
    assert AggregationMethod.COORD_MEDIAN is not None
    assert AggregationMethod.MULTI_KRUM is not None


@pytest.mark.skipif(not HAS_BYZANTINE, reason="byzantine_detector not available")
def test_anomaly_score_creation():
    """Test AnomalyScore dataclass."""
    score = AnomalyScore(node_id="node-1", score=0.5, is_byzantine=True, method="krum")
    assert score.node_id == "node-1"
    assert score.score == 0.5
    assert score.is_byzantine is True
    assert score.method == "krum"


@pytest.mark.skipif(
    not HAS_BYZANTINE or not HAS_TORCH, reason="byzantine_detector or torch not available"
)
def test_coordinate_median_aggregate_multiple():
    """Test CoordinateWiseMedian with multiple nodes."""
    det = CoordinateWiseMedian()
    updates = {
        "node-a": torch.tensor([1.0, 2.0, 3.0]),
        "node-b": torch.tensor([1.1, 2.1, 3.1]),
        "node-c": torch.tensor([0.9, 1.9, 2.9]),
    }
    out = det.aggregate(updates)
    assert out.shape == (3,)


@pytest.mark.skipif(
    not HAS_BYZANTINE or not HAS_TORCH, reason="byzantine_detector or torch not available"
)
def test_trimmed_mean_with_byzantine():
    """Test TrimmedMean filters byzantine nodes."""
    det = TrimmedMean(max_byzantine_fraction=0.3)
    # Include a byzantine node with extreme values
    updates = {
        "honest-1": torch.ones(10),
        "honest-2": torch.ones(10) * 1.05,
        "honest-3": torch.ones(10) * 0.95,
        "byzantine": torch.ones(10) * 100,  # Extreme outlier
    }
    out = det.aggregate(updates)
    # Result should be close to honest values, not the outlier
    assert torch.all(out < 10)


@pytest.mark.skipif(
    not HAS_BYZANTINE or not HAS_TORCH, reason="byzantine_detector or torch not available"
)
def test_krum_selects_best():
    """Test Krum selects the most representative gradient."""
    det = Krum(max_byzantine_fraction=0.3)
    # All honest nodes have similar values
    honest_grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    updates = {
        "honest-1": honest_grad,
        "honest-2": honest_grad + 0.1,
        "honest-3": honest_grad - 0.1,
        "byzantine": torch.tensor([100.0, -100.0, 100.0, -100.0]),
    }
    out = det.aggregate(updates)
    # Result should be similar to honest gradients
    assert abs(out[0].item() - 1.0) < 1.0


@pytest.mark.skipif(
    not HAS_BYZANTINE or not HAS_TORCH, reason="byzantine_detector or torch not available"
)
def test_multi_krum_selects_multiple():
    """Test MultiKrum selects multiple good candidates."""
    det = MultiKrum(max_byzantine_fraction=0.3, num_selected=2)
    honest_grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    updates = {
        "honest-1": honest_grad,
        "honest-2": honest_grad + 0.1,
        "honest-3": honest_grad - 0.1,
        "honest-4": honest_grad + 0.2,
        "byzantine": torch.tensor([100.0, -100.0, 100.0, -100.0]),
    }
    out = det.aggregate(updates)
    # Result should be similar to honest gradients (averaged)
    assert abs(out[0].item() - 1.0) < 1.0


@pytest.mark.skipif(
    not HAS_BYZANTINE or not HAS_TORCH, reason="byzantine_detector or torch not available"
)
def test_robust_aggregator_methods():
    """Test RobustAggregator with different methods."""
    for method in [
        AggregationMethod.TRIMMED_MEAN,
        AggregationMethod.COORD_MEDIAN,
        AggregationMethod.KRUM,
    ]:
        agg = RobustAggregator(method=method)
        updates = {
            "node-1": torch.ones(8),
            "node-2": torch.ones(8) * 1.05,
            "node-3": torch.ones(8) * 0.95,
        }
        out = agg.aggregate(updates)
        assert out.shape == (8,)
        # Should be close to average of honest nodes
        assert torch.all(out > 0.9) and torch.all(out < 1.1)


@pytest.mark.skipif(
    not HAS_BYZANTINE or not HAS_TORCH, reason="byzantine_detector or torch not available"
)
def test_detect_anomalies_returns_scores():
    """Test detect_anomalies returns proper scores."""
    det = TrimmedMean(max_byzantine_fraction=0.2)
    updates = {
        "normal-1": torch.ones(10),
        "normal-2": torch.ones(10) * 1.02,
        "normal-3": torch.ones(10) * 0.98,
        "outlier": torch.ones(10) * 5.0,
    }
    scores = det.detect_anomalies(updates)
    assert len(scores) == 4
    # Verify all scores are returned with proper structure
    for score in scores:
        assert isinstance(score.score, (int, float))
        assert score.score >= 0.0
