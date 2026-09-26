"""Base classes for Byzantine fault detection."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class AggregationMethod(Enum):
    """Aggregation methods for Byzantine fault tolerance."""

    COORD_MEDIAN = "coordinate_median"
    TRIMMED_MEAN = "trimmed_mean"
    KRUM = "krum"
    MULTI_KRUM = "multi_krum"
    CLUSTERING = "clustering"
    AUROR = "auror"
    ADAPTIVE = "adaptive"


@dataclass
class AnomalyScore:
    """Score indicating potential Byzantine behavior.

    Attributes:
        node_id: Identifier of the node.
        score: Anomaly score (higher = more suspicious).
        is_byzantine: Whether node is flagged as Byzantine.
        method: Detection method used.
        details: Additional diagnostic information.
    """

    node_id: str
    score: float
    is_byzantine: bool
    method: str
    details: dict[str, Any] = field(default_factory=dict)


class ByzantineDetector(ABC):
    """Abstract base class for Byzantine gradient detectors.

    All Byzantine detection methods must implement aggregate() and
    detect_anomalies() to provide both robust aggregation and
    anomaly detection capabilities.

    Attributes:
        max_byzantine_fraction: Maximum expected fraction of malicious nodes.
        device: PyTorch device for computations.
        node_reputation: Dict mapping node IDs to reputation scores.
    """

    def __init__(
        self,
        max_byzantine_fraction: float = 0.2,
        device: str = "cpu",
    ) -> None:
        self.max_byzantine_fraction = max_byzantine_fraction
        self.device = device
        self.node_reputation: dict[str, float] = {}

    @abstractmethod
    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate gradients from multiple nodes.

        Args:
            updates: Dictionary mapping node IDs to gradient tensors.

        Returns:
            Aggregated gradient tensor.
        """
        pass

    @abstractmethod
    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Detect anomalous nodes based on gradient updates.

        Args:
            updates: Dictionary mapping node IDs to gradient tensors.

        Returns:
            List of AnomalyScore objects for each node.
        """
        pass

    def update_reputation(self, node_id: str, delta: float) -> None:
        """Update reputation score for a node.

        Args:
            node_id: Node identifier.
            delta: Reputation change (positive or negative).
        """
        current = self.node_reputation.get(node_id, 0.5)
        self.node_reputation[node_id] = max(0.0, min(1.0, current + delta))
