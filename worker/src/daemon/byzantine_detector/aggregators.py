"""High-level aggregators combining multiple methods."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import torch

from .base import AggregationMethod, AnomalyScore, ByzantineDetector
from .bucketing import BucketingClipping
from .foolsgold import FoolsGoldFilter, NodeGradients
from .methods import (
    AUROR,
    ClusteringDetector,
    CoordinateWiseMedian,
    Krum,
    MultiKrum,
    TrimmedMean,
)
from .reputation import ReputationWeightedAggregator
from .signguard import SignGuard

logger = logging.getLogger(__name__)

DefenseFilter = Callable[[NodeGradients], NodeGradients]


def _tensor_updates_to_dict(
    updates: dict[str, torch.Tensor],
) -> tuple[NodeGradients, dict[str, torch.Tensor]]:
    """Convert ``{node_id: tensor}`` to the NodeGradients dict form.

    Returns the dict-of-layers form together with the original tensor map so
    the caller can restore tensor dtype/device after pipeline processing.
    """
    out: NodeGradients = {}
    for nid, t in updates.items():
        arr = t.detach().cpu().numpy()
        out[nid] = {"_flat": arr}
    return out, updates


def _dict_updates_to_tensor(
    dict_updates: NodeGradients,
    template: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Inverse of :func:`_tensor_updates_to_dict` matching the template tensors."""
    # Pick any template tensor for dtype/device/shape reference.
    ref = next(iter(template.values())) if template else None
    out: dict[str, torch.Tensor] = {}
    for nid, layers in dict_updates.items():
        if "_flat" in layers:
            arr = np.asarray(layers["_flat"])
        else:
            # Concatenate alphabetically (mirrors _flatten_node).
            keys = sorted(layers.keys())
            arr = (
                np.concatenate([np.asarray(layers[k]).ravel() for k in keys])
                if keys
                else np.zeros(0)
            )
        if ref is not None:
            tensor = torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
            if tensor.shape != ref.shape and arr.size == ref.numel():
                tensor = tensor.reshape(ref.shape)
        else:
            tensor = torch.as_tensor(arr)
        out[nid] = tensor
    return out


def default_defense_pipeline() -> list[DefenseFilter]:
    """Return the default ``[FoolsGoldFilter, SignGuard, BucketingClipping]``."""
    return [FoolsGoldFilter(), SignGuard(), BucketingClipping(bucket_size=2)]


class AdaptiveAggregator(ByzantineDetector):
    """Aggregator that auto-selects method based on node count.

    Uses different aggregation strategies based on cluster size:
    - n < 5: Coordinate-wise median (simple, robust)
    - n < 15: Multi-Krum (handles small clusters)
    - n >= 15: Clustering (scales to large clusters)

    Attributes:
        method_used: Last aggregation method selected.
    """

    def __init__(
        self,
        max_byzantine_fraction: float = 0.2,
        device: str = "cpu",
        defense_pipeline: list[DefenseFilter] | None = None,
    ) -> None:
        super().__init__(max_byzantine_fraction, device)

        # Initialize all methods
        self.median = CoordinateWiseMedian(max_byzantine_fraction, device)
        self.trimmed = TrimmedMean(max_byzantine_fraction, device)
        self.krum = Krum(max_byzantine_fraction, device)
        self.multikrum = MultiKrum(max_byzantine_fraction, device=device)
        self.clustering = ClusteringDetector(max_byzantine_fraction, device=device)
        self.auror = AUROR(max_byzantine_fraction, device=device)

        # Composable defense pipeline applied before base aggregation.
        # Default is empty -> bit-identical legacy behaviour. Pass
        # ``default_defense_pipeline()`` to opt into FoolsGold + SignGuard +
        # Bucketing. Each filter is expected to be a no-op below its own
        # internal min_nodes threshold.
        self.defense_pipeline: list[DefenseFilter] = (
            list(defense_pipeline) if defense_pipeline is not None else []
        )

        self.method_used: str | None = None
        self.last_pipeline_node_count: int | None = None

    def _apply_pipeline(self, updates: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run ``updates`` through the configured defense filters.

        Each filter operates on the ``NodeGradients`` dict form
        (``dict[str, dict[str, np.ndarray]]``) per the user-facing API; we
        convert at the boundary so existing tensor-based callers are
        unaffected. If the pipeline is empty the input is returned unchanged
        with no conversion overhead.
        """
        if not self.defense_pipeline:
            return updates
        if len(updates) < 4:
            # All defenses are no-ops below 4 nodes per spec.
            logger.debug(
                "AdaptiveAggregator: %d updates < 4, skipping defense pipeline",
                len(updates),
            )
            return updates
        dict_form, template = _tensor_updates_to_dict(updates)
        for f in self.defense_pipeline:
            dict_form = f(dict_form)
            if not dict_form:
                logger.warning(
                    "AdaptiveAggregator: defense pipeline %s emptied the "
                    "update set; falling back to original updates",
                    type(f).__name__,
                )
                return updates
        self.last_pipeline_node_count = len(dict_form)
        return _dict_updates_to_tensor(dict_form, template)

    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate using auto-selected method.

        If ``defense_pipeline`` was supplied at construction time the updates
        are pre-processed by each filter in order before being passed to the
        size-selected base aggregator. With an empty pipeline (the default)
        this is bit-identical to the previous implementation.
        """
        if not updates:
            return torch.tensor([])

        updates = self._apply_pipeline(updates)

        n = len(updates)

        if n < 5:
            self.method_used = "coordinate_median"
            return self.median.aggregate(updates)
        elif n < 15:
            self.method_used = "multi_krum"
            return self.multikrum.aggregate(updates)
        else:
            self.method_used = "clustering"
            return self.clustering.aggregate(updates)

    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Detect anomalies using selected method."""
        if self.method_used == "coordinate_median":
            return self.median.detect_anomalies(updates)
        elif self.method_used == "multi_krum":
            return self.multikrum.detect_anomalies(updates)
        elif self.method_used == "clustering":
            return self.clustering.detect_anomalies(updates)
        else:
            return self.clustering.detect_anomalies(updates)


class RobustAggregator(ByzantineDetector):
    """Production-ready aggregator with reputation tracking.

    Combines multiple Byzantine-robust methods with:
    - Reputation-based weighting for large clusters
    - Method selection for different cluster sizes
    - Statistics tracking for monitoring

    Attributes:
        method: Primary aggregation method.
        enable_reputation: Whether to use reputation weighting.
        aggregation_count: Number of aggregations performed.
        detected_byzantine_count: Total Byzantine nodes detected.
    """

    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.MULTI_KRUM,
        max_byzantine_fraction: float = 0.2,
        device: str = "cpu",
        enable_reputation: bool = True,
        enable_history: bool = True,
    ) -> None:
        self.method = method
        self.max_byzantine_fraction = max_byzantine_fraction
        self.device = device
        self.enable_reputation = enable_reputation
        self.enable_history = enable_history

        # Initialize primary detector
        if method == AggregationMethod.COORD_MEDIAN:
            self.detector = CoordinateWiseMedian(max_byzantine_fraction, device)
        elif method == AggregationMethod.TRIMMED_MEAN:
            self.detector = TrimmedMean(max_byzantine_fraction, device)
        elif method == AggregationMethod.KRUM:
            self.detector = Krum(max_byzantine_fraction, device)
        elif method == AggregationMethod.MULTI_KRUM:
            self.detector = MultiKrum(max_byzantine_fraction, device=device)
        elif method == AggregationMethod.CLUSTERING:
            self.detector = ClusteringDetector(max_byzantine_fraction, device=device)
        elif method == AggregationMethod.AUROR:
            self.detector = AUROR(max_byzantine_fraction, device=device)
        elif method == AggregationMethod.ADAPTIVE:
            self.detector = AdaptiveAggregator(max_byzantine_fraction, device)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Add reputation-based aggregator if enabled
        if enable_reputation:
            self.reputation_aggregator = ReputationWeightedAggregator(
                max_byzantine_fraction, device
            )

        self.aggregation_count = 0
        self.detected_byzantine_count = 0

    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate with optional reputation weighting."""
        self.aggregation_count += 1

        # Use reputation-based for large sets
        if self.enable_reputation and len(updates) >= 20:
            return self.reputation_aggregator.aggregate(updates)

        return self.detector.aggregate(updates)

    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Detect anomalies with enhanced tracking."""
        if self.enable_reputation and len(updates) >= 10:
            scores = self.reputation_aggregator.detect_anomalies(updates)
        else:
            scores = self.detector.detect_anomalies(updates)

        # Track statistics
        for score in scores:
            if score.is_byzantine:
                self.detected_byzantine_count += 1

        return scores

    def get_node_reputation(self, node_id: str) -> float:
        """Get reputation score for a node."""
        if hasattr(self, "reputation_aggregator"):
            profile = self.reputation_aggregator.get_or_create_profile(node_id)
            return profile.reputation_score
        return self.detector.node_reputation.get(node_id, 0.5)

    def get_node_history(self, node_id: str) -> list | None:
        """Get gradient history for a node."""
        if hasattr(self, "reputation_aggregator"):
            profile = self.reputation_aggregator.get_or_create_profile(node_id)
            return profile.gradient_history
        return None

    def filter_by_reputation(self, updates: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Filter updates to include only reputable nodes."""
        reputation_threshold = 0.3

        if hasattr(self, "reputation_aggregator"):
            return {
                nid: upd
                for nid, upd in updates.items()
                if self.reputation_aggregator.get_or_create_profile(nid).reputation_score
                >= reputation_threshold
            }

        return {
            nid: upd
            for nid, upd in updates.items()
            if self.detector.node_reputation.get(nid, 0.5) >= reputation_threshold
        }

    def get_stats(self) -> dict:
        """Get aggregator statistics."""
        stats = {
            "method": self.method.value,
            "aggregation_count": self.aggregation_count,
            "detected_byzantine_total": self.detected_byzantine_count,
            "max_byzantine_fraction": self.max_byzantine_fraction,
            "reputation_enabled": self.enable_reputation,
            "history_enabled": self.enable_history,
        }

        if hasattr(self, "reputation_aggregator"):
            profiles = self.reputation_aggregator.node_profiles
            stats["nodes_tracked"] = len(profiles)
            if profiles:
                avg_rep = sum(p.reputation_score for p in profiles.values())
                avg_rep /= len(profiles)
                stats["avg_reputation"] = avg_rep
                stats["flagged_nodes"] = sum(1 for p in profiles.values() if p.flagged_count > 0)

        return stats
