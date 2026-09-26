"""SignGuard sign-statistics defense for federated learning."""

from __future__ import annotations

import logging

import numpy as np

from .foolsgold import NodeGradients, _stack_nodes

logger = logging.getLogger(__name__)


class SignGuard:
    """SignGuard sign-statistics defense (arxiv:2109.05872).

    Defends against sign-flipping and value-manipulation attacks by checking,
    per coordinate, whether each node's gradient sign agrees with the
    elementwise median sign across all nodes. Nodes whose agreement score is
    a statistical outlier (low) on the negative tail of a z-score
    distribution are flagged as Byzantine. The original paper uses a feature
    cluster (MeanShift / DBSCAN); this implementation uses a one-sided
    z-score filter that is parameter-free and avoids the sklearn dependency
    while preserving the same flag-the-low-agreement-outliers behaviour.

    Pure-NumPy implementation operating on
    ``dict[str, dict[str, np.ndarray]]`` (node_id -> layer_name -> ndarray).

    Attributes:
        z_threshold: A node is flagged when its modified-z-score (computed
            from the median + median-absolute-deviation) is more than
            ``z_threshold`` units below the median AND its agreement falls
            below ``median - abs_floor``. Using median + MAD instead of
            mean + std makes the detector robust to attackers contaminating
            the mean/std estimates. Default 3.5 follows the Iglewicz-Hoaglin
            rule of thumb for modified z-score outlier flagging.
        abs_floor: Minimum absolute drop from the median agreement score
            required before flagging is honoured. Prevents false-positive
            flags when honest nodes are so well aligned that the MAD is
            essentially noise. Default 0.05 (5pp) is conservative for the
            typical 30-90% honest-agreement regime.
        min_nodes: Minimum cluster size; below this the filter is a no-op.
    """

    def __init__(
        self,
        z_threshold: float = 3.5,
        abs_floor: float = 0.05,
        min_nodes: int = 4,
    ) -> None:
        self.z_threshold = float(z_threshold)
        self.abs_floor = float(abs_floor)
        self.min_nodes = int(min_nodes)
        self.last_scores: dict[str, float] = {}
        self.last_flagged: set[str] = set()

    def _agreement_scores(self, node_gradients: NodeGradients) -> tuple[list[str], np.ndarray]:
        """Compute per-node sign-agreement scores against the median sign.

        Args:
            node_gradients: Mapping of node_id to per-layer gradient ndarrays.

        Returns:
            Tuple ``(ordered_node_ids, scores)`` where ``scores[i]`` is the
            fraction of coordinates whose sign matches the elementwise median
            sign across all nodes. Zero-magnitude coordinates contribute
            sign 0, which trivially matches any other zero coordinate.
        """
        ids, mat = _stack_nodes(node_gradients)
        n, d = mat.shape
        if n == 0 or d == 0:
            return ids, np.zeros(n, dtype=np.float64)

        signs = np.sign(mat)  # (n, d) in {-1, 0, +1}
        median_signs = np.sign(np.median(mat, axis=0))  # (d,)
        agreement = (signs == median_signs[None, :]).astype(np.float64)
        scores = agreement.mean(axis=1)
        return ids, scores

    def filter(self, node_gradients: NodeGradients) -> tuple[NodeGradients, set[str]]:
        """Drop nodes whose sign-agreement score is a low outlier.

        Args:
            node_gradients: Mapping of node_id to per-layer gradient ndarrays.

        Returns:
            ``(kept, flagged)`` where ``kept`` is a filtered copy of the
            input with flagged nodes removed (sharing the original ndarrays
            without copy) and ``flagged`` is the set of removed node ids.
        """
        if len(node_gradients) < self.min_nodes:
            logger.debug(
                "SignGuard: %d nodes < min_nodes=%d, skipping",
                len(node_gradients),
                self.min_nodes,
            )
            self.last_scores = dict.fromkeys(node_gradients, 1.0)
            self.last_flagged = set()
            return node_gradients, set()

        ids, scores = self._agreement_scores(node_gradients)
        med = float(np.median(scores))
        # Median absolute deviation, scaled by 1.4826 (Gaussian-consistent).
        mad = 1.4826 * float(np.median(np.abs(scores - med)))

        flagged: set[str] = set()
        if mad < 1e-12:
            # Degenerate MAD (most nodes sit exactly at the median). Fall
            # back to pure absolute-floor flagging: any node that scores at
            # least `abs_floor` below the median is suspicious. This handles
            # the binary-cluster failure mode where 80%+ of scores collide
            # at 1.0 and MAD collapses to zero.
            if float(np.max(scores) - np.min(scores)) < self.abs_floor:
                self.last_scores = {nid: float(s) for nid, s in zip(ids, scores, strict=True)}
                self.last_flagged = set()
                return node_gradients, set()
            for nid, si in zip(ids, scores, strict=True):
                if (med - si) >= self.abs_floor:
                    flagged.add(nid)
        else:
            modified_z = (scores - med) / mad
            for nid, zi, si in zip(ids, modified_z, scores, strict=True):
                # One-sided: only LOW agreement is suspicious. High
                # agreement means the node tracks the majority direction.
                # The abs_floor guard prevents false positives when MAD is
                # essentially noise.
                if zi < -self.z_threshold and (med - si) >= self.abs_floor:
                    flagged.add(nid)

        # Guardrail: never flag the entire cluster. If somehow all nodes are
        # flagged (degenerate case), return everything kept.
        if len(flagged) >= len(ids):
            logger.warning(
                "SignGuard: would flag all %d nodes; treating as no-op",
                len(ids),
            )
            flagged = set()

        kept = {nid: g for nid, g in node_gradients.items() if nid not in flagged}
        self.last_scores = {nid: float(s) for nid, s in zip(ids, scores, strict=True)}
        self.last_flagged = flagged
        if flagged:
            logger.debug(
                "SignGuard flagged %d/%d nodes: %s",
                len(flagged),
                len(ids),
                sorted(flagged),
            )
        return kept, flagged

    def __call__(self, node_gradients: NodeGradients) -> NodeGradients:
        """Pipeline-friendly: return only the kept gradients."""
        kept, _ = self.filter(node_gradients)
        return kept
