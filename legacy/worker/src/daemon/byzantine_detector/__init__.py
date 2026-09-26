"""Byzantine fault detection and robust aggregation.

This package provides multiple Byzantine-robust aggregation methods
for federated learning, including:

- Coordinate-wise median
- Trimmed mean
- Krum and Multi-Krum
- Clustering-based detection
- AUROR (iterative reweighting)
- Adaptive aggregator (auto-selects method)
- Reputation-weighted aggregator

Example:
    >>> from worker.src.daemon.byzantine_detector import RobustAggregator
    >>> aggregator = RobustAggregator(method=AggregationMethod.MULTI_KRUM)
    >>> result = aggregator.aggregate(updates)
"""

from .aggregators import (
    AdaptiveAggregator,
    RobustAggregator,
    default_defense_pipeline,
)
from .base import AggregationMethod, AnomalyScore, ByzantineDetector
from .bucketing import BucketingClipping
from .foolsgold import FoolsGold, FoolsGoldFilter
from .methods import (
    AUROR,
    ClusteringDetector,
    CoordinateWiseMedian,
    Krum,
    MultiKrum,
    TrimmedMean,
)
from .reputation import (
    GradientHistoryEntry,
    NodeBehaviorProfile,
    ReputationWeightedAggregator,
)
from .signguard import SignGuard

__all__ = [
    # Base classes
    "ByzantineDetector",
    "AggregationMethod",
    "AnomalyScore",
    # Methods
    "CoordinateWiseMedian",
    "TrimmedMean",
    "Krum",
    "MultiKrum",
    "ClusteringDetector",
    "AUROR",
    # Advanced
    "AdaptiveAggregator",
    "RobustAggregator",
    "ReputationWeightedAggregator",
    "GradientHistoryEntry",
    "NodeBehaviorProfile",
    # v1.1 Byzantine defenses
    "FoolsGold",
    "FoolsGoldFilter",
    "SignGuard",
    "BucketingClipping",
    "default_defense_pipeline",
]
