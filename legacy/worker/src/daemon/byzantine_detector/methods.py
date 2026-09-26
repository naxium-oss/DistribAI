"""Individual Byzantine detection methods."""

import numpy as np
import torch

from .base import AnomalyScore, ByzantineDetector


class CoordinateWiseMedian(ByzantineDetector):
    """Coordinate-wise median aggregation.

    Takes the median of each gradient dimension independently.
    Robust to up to 50% Byzantine nodes but loses gradient magnitude info.
    """

    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate using coordinate-wise median."""
        if not updates:
            return torch.tensor([])
        stacked = torch.stack(list(updates.values()))
        median = torch.median(stacked, dim=0).values
        return median

    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Detect anomalies using distance from median."""
        if not updates:
            return []

        median = self.aggregate(updates)
        if median.numel() == 0:
            return [AnomalyScore(nid, 0.0, False, "coordinate_median") for nid in updates]

        dists = [torch.norm(t.to(self.device) - median).item() for t in updates.values()]
        max_d = max(dists) if dists else 0.0
        thresh = max_d * 0.75 if max_d > 0 else 0.0

        return [
            AnomalyScore(nid, d, d > thresh, "coordinate_median")
            for nid, d in zip(updates.keys(), dists)
        ]


class TrimmedMean(ByzantineDetector):
    """Trimmed mean aggregation.

    Removes top and bottom fraction of values before averaging.
    Robust to the trimmed fraction of Byzantine nodes.
    """

    def __init__(
        self,
        max_byzantine_fraction: float = 0.2,
        device: str = "cpu",
    ) -> None:
        super().__init__(max_byzantine_fraction, device)
        self.trim_fraction = max_byzantine_fraction

    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate using trimmed mean."""
        if not updates:
            return torch.tensor([])

        stacked = torch.stack(list(updates.values())).to(self.device)
        n = stacked.shape[0]
        k = int(n * self.trim_fraction)

        if k == 0:
            return torch.mean(stacked, dim=0)

        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[k : n - k, ...]
        return torch.mean(trimmed, dim=0)

    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Detect anomalies using trimming position."""
        if not updates:
            return []

        stacked = torch.stack(list(updates.values())).to(self.device)
        n = stacked.shape[0]
        k = int(self.trim_fraction * n)

        if k == 0:
            return [AnomalyScore(nid, 0.0, False, "trimmed_mean") for nid in updates]

        sorted_vals, _ = torch.sort(stacked, dim=0)
        scores = []

        for i, node_id in enumerate(updates.keys()):
            is_trimmed = (i < k) or (i >= n - k)
            score = 1.0 if is_trimmed else 0.0
            scores.append(AnomalyScore(node_id, score, is_trimmed, "trimmed_mean"))

        return scores


class Krum(ByzantineDetector):
    """Krum aggregation (single best gradient).

    Selects the gradient with smallest sum of distances to its neighbors.
    Robust to less than n/2 Byzantine nodes.
    """

    def __init__(
        self,
        max_byzantine_fraction: float = 0.2,
        device: str = "cpu",
    ) -> None:
        super().__init__(max_byzantine_fraction, device)
        self.num_byzantine = int(max_byzantine_fraction * 100)

    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate using Krum (select single best)."""
        if not updates:
            return torch.tensor([])

        stacked = torch.stack(list(updates.values())).to(self.device)
        n = stacked.shape[0]
        num_neighbors = n - self.num_byzantine - 2

        if num_neighbors < 1:
            num_neighbors = 1

        distances = self._compute_distances(stacked)
        scores = []

        for i in range(n):
            node_distances = distances[i]
            sorted_d, _ = torch.sort(node_distances)
            score = torch.sum(sorted_d[:num_neighbors])
            scores.append(score)

        min_idx = torch.argmin(torch.tensor(scores))
        return stacked[min_idx]

    def _compute_distances(self, stacked: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances between gradients."""
        n = stacked.shape[0]
        distances = torch.zeros(n, n, device=self.device)

        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = stacked[i] - stacked[j]
                    distances[i, j] = torch.norm(diff)

        return distances

    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Detect anomalies using Krum scores."""
        if not updates:
            return []

        node_ids = list(updates.keys())
        stacked = torch.stack(list(updates.values())).to(self.device)
        n = stacked.shape[0]

        num_neighbors = n - self.num_byzantine - 2
        if num_neighbors < 1:
            num_neighbors = 1

        distances = self._compute_distances(stacked)
        scores = []

        for i in range(n):
            node_distances = distances[i]
            sorted_d, _ = torch.sort(node_distances)
            score = torch.sum(sorted_d[:num_neighbors])
            scores.append(score.item())

        max_score = max(scores) if scores else 1.0
        if max_score == 0:
            max_score = 1.0

        normalized = [s / max_score for s in scores]
        threshold = np.percentile(normalized, 75)

        return [
            AnomalyScore(nid, normalized[i], normalized[i] > threshold, "krum")
            for i, nid in enumerate(node_ids)
        ]


class MultiKrum(ByzantineDetector):
    """Multi-Krum aggregation (average of top-k best gradients).

    Averages the m gradients with smallest Krum scores.
    More robust than single Krum for high-dimensional gradients.
    """

    def __init__(
        self,
        max_byzantine_fraction: float = 0.2,
        num_selected: int = 5,
        device: str = "cpu",
    ) -> None:
        super().__init__(max_byzantine_fraction, device)
        self.num_selected = num_selected
        self.krum = Krum(max_byzantine_fraction, device)

    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate using Multi-Krum."""
        if not updates:
            return torch.tensor([])

        stacked = torch.stack(list(updates.values())).to(self.device)
        n = stacked.shape[0]
        num_selected = min(self.num_selected, n)

        distances = self.krum._compute_distances(stacked)
        num_neighbors = n - self.krum.num_byzantine - 2

        if num_neighbors < 1:
            num_neighbors = 1

        scores = []
        for i in range(n):
            node_distances = distances[i]
            sorted_d, _ = torch.sort(node_distances)
            score = torch.sum(sorted_d[:num_neighbors])
            scores.append(score)

        sorted_indices = torch.argsort(torch.tensor(scores))[:num_selected]
        selected = stacked[sorted_indices]
        return torch.mean(selected, dim=0)

    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Use Krum for anomaly detection."""
        return self.krum.detect_anomalies(updates)


class ClusteringDetector(ByzantineDetector):
    """Clustering-based Byzantine detection.

    Uses z-score outlier detection to identify suspicious gradients.
    Robust to clusters of Byzantine nodes.
    """

    def __init__(
        self,
        max_byzantine_fraction: float = 0.2,
        threshold: float = 2.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(max_byzantine_fraction, device)
        self.threshold = threshold

    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate excluding outliers."""
        if not updates:
            return torch.tensor([])

        stacked = torch.stack(list(updates.values())).to(self.device)
        n = stacked.shape[0]

        mean = torch.mean(stacked, dim=0)
        std = torch.std(stacked, dim=0) + 1e-8
        z_scores = torch.norm((stacked - mean) / std, dim=1)
        median_z = torch.median(z_scores)

        suspicious_indices = (z_scores > median_z + self.threshold).nonzero(as_tuple=True)[0]

        if len(suspicious_indices) < n - 1:
            clean_indices = [i for i in range(n) if i not in suspicious_indices]
            clean_updates = stacked[clean_indices]
            aggregated = torch.mean(clean_updates, dim=0)
        else:
            aggregated = torch.median(stacked, dim=0).values

        return aggregated

    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Detect anomalies using z-scores."""
        if not updates:
            return []

        node_ids = list(updates.keys())
        stacked = torch.stack(list(updates.values())).to(self.device)

        mean = torch.mean(stacked, dim=0)
        std = torch.std(stacked, dim=0) + 1e-8
        z_scores = torch.norm((stacked - mean) / std, dim=1)
        median_z = torch.median(z_scores)

        scores = []
        for i, node_id in enumerate(node_ids):
            is_suspicious = z_scores[i] > median_z + self.threshold
            score = ((z_scores[i] - median_z) / self.threshold).item()
            scores.append(AnomalyScore(node_id, score, is_suspicious, "clustering"))

        return scores


class AUROR(ByzantineDetector):
    """AUROR: Adaptive Weighted Aggregation.

    Iteratively reweights gradients based on distance from current estimate.
    Robust to adaptive Byzantine attacks.
    """

    def __init__(
        self,
        max_byzantine_fraction: float = 0.2,
        num_iterations: int = 3,
        device: str = "cpu",
    ) -> None:
        super().__init__(max_byzantine_fraction, device)
        self.num_iterations = num_iterations

    def aggregate(self, updates: dict[str, torch.Tensor]) -> torch.Tensor:
        """Aggregate using iterative reweighting."""
        if not updates:
            return torch.tensor([])

        stacked = torch.stack(list(updates.values())).to(self.device)
        estimate = torch.mean(stacked, dim=0)

        for _ in range(self.num_iterations):
            distances = torch.norm(stacked - estimate, dim=1)
            weights = 1.0 / (distances + 1e-8)
            weights = weights / torch.sum(weights)
            estimate = torch.sum(stacked * weights.unsqueeze(-1), dim=0)

        return estimate

    def detect_anomalies(self, updates: dict[str, torch.Tensor]) -> list[AnomalyScore]:
        """Detect anomalies using final weights."""
        if not updates:
            return []

        node_ids = list(updates.keys())
        stacked = torch.stack(list(updates.values())).to(self.device)
        estimate = self.aggregate(updates)

        distances = torch.norm(stacked - estimate, dim=1)
        max_dist = torch.max(distances)
        normalized = (distances / max_dist).cpu().tolist()
        threshold = np.percentile(normalized, 75)

        return [
            AnomalyScore(nid, normalized[i], normalized[i] > threshold, "auror")
            for i, nid in enumerate(node_ids)
        ]
