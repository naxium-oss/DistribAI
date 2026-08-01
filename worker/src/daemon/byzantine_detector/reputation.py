"""Reputation-based Byzantine detection with gradient history."""

import hashlib
import time
from dataclasses import dataclass, field

import torch

from .base import AnomalyScore, ByzantineDetector
from .methods import ClusteringDetector


@dataclass
class GradientHistoryEntry:
    """Single gradient history entry for a node.

    Attributes:
        gradient_hash: Hash of gradient (not full tensor for memory).
        timestamp: When gradient was received.
        norm: L2 norm of gradient.
        loss: Optional loss value.
    """

    gradient_hash: str
    timestamp: float
    norm: float
    loss: float | None = None


@dataclass
class NodeBehaviorProfile:
    """Behavioral profile for detecting gradual attacks.

    Tracks gradient history and computes consistency scores
    to detect turncoat behavior (honest → malicious).

    Attributes:
        node_id: Node identifier.
        gradient_history: Recent gradient entries.
        reputation_score: Current reputation (0-1).
        flagged_count: Times flagged as Byzantine.
        last_flagged: Timestamp of last flagging.
        consistency_score: Gradient consistency metric.
    """

    node_id: str
    gradient_history: list[GradientHistoryEntry] = field(default_factory=list)
    reputation_score: float = 0.5
    flagged_count: int = 0
    last_flagged: float | None = None
    consistency_score: float = 1.0
    contribution_quality: float = 1.0

    def add_gradient(self, gradient: torch.Tensor, loss: float | None = None) -> None:
        """Add gradient to history."""
        grad_bytes = gradient.cpu().numpy().tobytes()
        gradient_hash = hashlib.sha256(grad_bytes).hexdigest()[:16]

        entry = GradientHistoryEntry(
            gradient_hash=gradient_hash,
            timestamp=time.time(),
            norm=torch.norm(gradient).item(),
            loss=loss,
        )
        self.gradient_history.append(entry)

        # Keep only last 100 entries
        if len(self.gradient_history) > 100:
            self.gradient_history = self.gradient_history[-100:]

    def get_gradient_consistency(self) -> float:
        """Calculate gradient consistency score.

        Lower variance in gradient norms = higher consistency.

        Returns:
            Consistency score (0-1, higher = more consistent).
        """
        if len(self.gradient_history) < 10:
            return 1.0

        norms = [e.norm for e in self.gradient_history[-20:]]
        if not norms:
            return 1.0

        mean_norm = sum(norms) / len(norms)
        variance = sum((n - mean_norm) ** 2 for n in norms) / len(norms)

        consistency = 1.0 / (1.0 + variance / (mean_norm**2 + 1e-8))
        return max(0.0, min(1.0, consistency))

    def detect_turncoat_behavior(self) -> bool:
        """Detect nodes switching from honest to malicious.

        Compares early vs late gradient norms for sudden changes.

        Returns:
            True if turncoat behavior detected.
        """
        if len(self.gradient_history) < 20:
            return False

        mid = len(self.gradient_history) // 2
        early_norms = [e.norm for e in self.gradient_history[:mid]]
        late_norms = [e.norm for e in self.gradient_history[mid:]]

        early_mean = sum(early_norms) / len(early_norms) if early_norms else 1.0
        late_mean = sum(late_norms) / len(late_norms) if late_norms else 1.0

        ratio = late_mean / (early_mean + 1e-8)
        return ratio > 10.0 or ratio < 0.1


class ReputationWeightedAggregator(ByzantineDetector):
    """Aggregator that weights gradients by node reputation.

    Higher reputation nodes have more influence on the aggregate.
    Malicious nodes are automatically downweighted or excluded.

    Attributes:
        reputation_threshold: Minimum reputation to participate.
        min_weight: Minimum weight even for low-reputation nodes.
        node_profiles: Dict of behavior profiles per node.
    """

    def __init__(
        self,
        max_byzantine_fraction: float = 0.2,
        device: str = "cpu",
        reputation_threshold: float = 0.3,
        min_weight: float = 0.1,
    ) -> None:
        super().__init__(max_byzantine_fraction, device)
        self.reputation_threshold = reputation_threshold
        self.min_weight = min_weight
        self.node_profiles: dict[str, NodeBehaviorProfile] = {}

    def get_or_create_profile(self, node_id: str) -> NodeBehaviorProfile:
        """Get or create node behavior profile."""
        if node_id not in self.node_profiles:
            self.node_profiles[node_id] = NodeBehaviorProfile(node_id=node_id)
        return self.node_profiles[node_id]

    def compute_weights(self, updates: dict[str, torch.Tensor]) -> dict[str, float]:
        """Compute reputation-based weights for each node."""
        weights = {}

        for node_id in updates.keys():
            profile = self.get_or_create_profile(node_id)
            base_weight = max(self.min_weight, profile.reputation_score)
            consistency = profile.get_gradient_consistency()
            weight = base_weight * consistency
            weights[node_id] = weight

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        return weights

    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate with reputation weighting."""
        if not updates:
            return torch.tensor([])

        # Filter low-reputation nodes
        filtered = {
            nid: upd
            for nid, upd in updates.items()
            if self.get_or_create_profile(nid).reputation_score >= self.reputation_threshold
        }

        if not filtered:
            # Fall back to median if all nodes filtered
            from .methods import CoordinateWiseMedian

            median_agg = CoordinateWiseMedian(self.max_byzantine_fraction, self.device)
            return median_agg.aggregate(updates)

        # Compute and apply weights
        weights = self.compute_weights(filtered)
        stacked = torch.stack(list(filtered.values())).to(self.device)
        weight_tensor = torch.tensor(
            [weights[nid] for nid in filtered.keys()], device=self.device
        ).view(-1, 1)

        return torch.sum(stacked * weight_tensor, dim=0)

    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Detect anomalies with history-aware detection."""
        if not updates:
            return []

        # Use clustering detection as base
        detector = ClusteringDetector(self.max_byzantine_fraction, device=self.device)
        base_scores = detector.detect_anomalies(updates)
        base_scores_dict = {s.node_id: s for s in base_scores}

        scores = []
        for node_id, gradient in updates.items():
            profile = self.get_or_create_profile(node_id)
            profile.add_gradient(gradient)

            base_score = base_scores_dict.get(node_id)
            if base_score is None:
                base_score = AnomalyScore(node_id, 0.0, False, "reputation")

            # Check for turncoat behavior
            is_turncoat = profile.detect_turncoat_behavior()
            consistency = profile.get_gradient_consistency()

            # Combine scores
            anomaly_score = base_score.score
            if is_turncoat:
                anomaly_score += 0.5

            # Adjust for consistency
            anomaly_score *= (2.0 - consistency) / 2.0
            is_byzantine = anomaly_score > 0.75 or is_turncoat

            scores.append(
                AnomalyScore(
                    node_id=node_id,
                    score=anomaly_score,
                    is_byzantine=is_byzantine,
                    method="reputation_aware",
                    details={
                        "base_score": base_score.score,
                        "is_turncoat": is_turncoat,
                        "consistency": consistency,
                        "reputation": profile.reputation_score,
                    },
                )
            )

            # Update reputation
            if is_byzantine:
                profile.reputation_score *= 0.8
                profile.flagged_count += 1
                profile.last_flagged = time.time()
            else:
                # Gradually restore reputation
                profile.reputation_score = min(1.0, profile.reputation_score * 1.02 + 0.01)

        return scores
