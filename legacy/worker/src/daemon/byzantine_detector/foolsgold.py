"""FoolsGold sybil-similarity defense for federated learning."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


NodeGradients = dict[str, dict[str, np.ndarray]]
"""Mapping of node_id -> {layer_name -> np.ndarray}."""


def _flatten_node(grads: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate all layer arrays for a single node into one 1-D vector.

    Args:
        grads: Mapping of layer name to gradient ndarray.

    Returns:
        1-D float64 vector formed by ravelling each layer in sorted-key order.
    """
    if not grads:
        return np.zeros(0, dtype=np.float64)
    keys = sorted(grads.keys())
    parts = [np.asarray(grads[k], dtype=np.float64).ravel() for k in keys]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _stack_nodes(node_gradients: NodeGradients) -> tuple[list[str], np.ndarray]:
    """Stack per-node flattened gradients into a (n, d) matrix.

    Args:
        node_gradients: Mapping of node_id to per-layer gradient dict.

    Returns:
        Tuple of (ordered_node_ids, matrix). All node vectors must have the
        same length; ragged inputs are zero-padded to the max length and a
        warning is emitted.
    """
    ids = list(node_gradients.keys())
    vecs = [_flatten_node(node_gradients[nid]) for nid in ids]
    if not vecs:
        return ids, np.zeros((0, 0), dtype=np.float64)
    lengths = {v.shape[0] for v in vecs}
    if len(lengths) > 1:
        max_len = max(lengths)
        logger.warning(
            "FoolsGold: ragged node gradients (lengths=%s); zero-padding to %d",
            sorted(lengths),
            max_len,
        )
        padded = np.zeros((len(vecs), max_len), dtype=np.float64)
        for i, v in enumerate(vecs):
            padded[i, : v.shape[0]] = v
        return ids, padded
    return ids, np.stack(vecs, axis=0)


class FoolsGold:
    """FoolsGold sybil-similarity defense (arxiv:1808.04866).

    Mitigates sybil-based gradient poisoning by penalising nodes whose
    flattened gradient vectors are abnormally similar to other nodes' (a
    hallmark of attackers using a shared adversarial objective). For each
    node we compute the maximum cosine similarity to any other node, scale
    by the maximum max-similarity across nodes (the "pardoning" step), then
    optionally apply the logit transform ``alpha_i = log((1 - s_i)/s_i) + 0.5``
    clipped to ``[0, 1]`` (Fung et al. 2018).

    Pure-NumPy implementation operating on
    ``dict[str, dict[str, np.ndarray]]`` (node_id -> layer_name -> ndarray).

    Attributes:
        epsilon: Floor added to vector norms and logit denominators to keep
            cosine math and the log transform numerically stable.
        use_logit: If True, apply the Fung et al. logit reweighting on top of
            the linear ``1 - s`` weights.
        pardoning: If True, divide each max-similarity by the global maximum
            max-similarity so that the most-similar honest pair is not
            penalised (Fung et al. equation 4).
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        use_logit: bool = True,
        pardoning: bool = True,
    ) -> None:
        self.epsilon = float(epsilon)
        self.use_logit = bool(use_logit)
        self.pardoning = bool(pardoning)

    def compute_weights(self, node_gradients: NodeGradients) -> dict[str, float]:
        """Compute FoolsGold per-node weight multipliers in ``[0, 1]``.

        Args:
            node_gradients: Mapping of node_id to per-layer gradient ndarrays.

        Returns:
            Dictionary mapping each node_id to a weight in ``[0, 1]``. Returns
            a uniform 1.0 vector when fewer than two nodes are supplied, since
            sybil detection is meaningless without peers to compare against.
        """
        ids, mat = _stack_nodes(node_gradients)
        n = mat.shape[0]
        if n < 2:
            # Cannot compute pairwise similarities; pass through unchanged.
            return dict.fromkeys(ids, 1.0)

        # Cosine similarity matrix. Adding epsilon to the norm avoids the
        # near-zero-gradient instability (see numerical-stability note below).
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        normed = mat / (norms + self.epsilon)
        sim = normed @ normed.T  # (n, n)
        np.fill_diagonal(sim, -np.inf)

        # max similarity each node has with ANY other node
        max_sim = np.max(sim, axis=1)  # (n,)
        # Replace any remaining -inf (n==1 edge case already handled above)
        max_sim = np.clip(max_sim, 0.0, 1.0)

        # Pardoning: rescale so the maximum max-similarity is 1.
        if self.pardoning:
            global_max = float(max_sim.max())
            if global_max > self.epsilon:
                max_sim = max_sim / global_max

        # Linear weights in [0, 1].
        weights = 1.0 - max_sim
        weights = np.clip(weights, 0.0, 1.0)

        if self.use_logit:
            # alpha = log((1 - s) / s) + 0.5, clipped to [0, 1].
            # Bound s away from 0 and 1 to keep the log finite.
            s = np.clip(max_sim, self.epsilon, 1.0 - self.epsilon)
            alpha = np.log((1.0 - s) / s) + 0.5
            # Sigmoid-like clip per Fung et al. eqn 6.
            weights = np.clip(alpha, 0.0, 1.0)

        return {nid: float(w) for nid, w in zip(ids, weights, strict=True)}


class FoolsGoldFilter:
    """Filter wrapper that pre-weights gradients via FoolsGold (arxiv:1808.04866).

    Applies :class:`FoolsGold` weights elementwise to each layer of each node
    before handing the gradients to a downstream aggregator. Designed to plug
    into :class:`~worker.src.daemon.byzantine_detector.aggregators.AdaptiveAggregator`'s
    ``defense_pipeline``.

    The filter is a no-op when fewer than ``min_nodes`` nodes are present
    (default 4) so that sybil detection is only applied when the cluster is
    large enough for the cosine-similarity statistic to be meaningful.

    Attributes:
        min_nodes: Threshold below which the filter passes gradients through
            unchanged.
        detector: Underlying :class:`FoolsGold` instance.
    """

    def __init__(
        self,
        min_nodes: int = 4,
        epsilon: float = 1e-8,
        use_logit: bool = True,
        pardoning: bool = True,
    ) -> None:
        self.min_nodes = int(min_nodes)
        self.detector = FoolsGold(epsilon=epsilon, use_logit=use_logit, pardoning=pardoning)
        self.last_weights: dict[str, float] = {}

    def __call__(self, node_gradients: NodeGradients) -> NodeGradients:
        """Reweight each node's gradients in place-clone."""
        if len(node_gradients) < self.min_nodes:
            logger.debug(
                "FoolsGoldFilter: %d nodes < min_nodes=%d, skipping",
                len(node_gradients),
                self.min_nodes,
            )
            self.last_weights = dict.fromkeys(node_gradients, 1.0)
            return node_gradients

        weights = self.detector.compute_weights(node_gradients)
        self.last_weights = weights
        out: NodeGradients = {}
        for nid, layers in node_gradients.items():
            w = weights.get(nid, 1.0)
            if w == 1.0:
                out[nid] = layers
                continue
            out[nid] = {k: (np.asarray(v) * w) for k, v in layers.items()}
        logger.debug(
            "FoolsGoldFilter weights: min=%.3f max=%.3f mean=%.3f",
            min(weights.values()) if weights else 1.0,
            max(weights.values()) if weights else 1.0,
            float(np.mean(list(weights.values()))) if weights else 1.0,
        )
        return out


# Convenience type alias for the AdaptiveAggregator defense_pipeline API.
DefenseFilter = Callable[[NodeGradients], NodeGradients]


def _coerce_to_dict_form(updates: Any) -> NodeGradients | None:
    """Best-effort coerce a torch-tensor or array input to the dict form.

    Returns ``None`` if the input cannot be interpreted (so callers can fall
    back to passing the input through unchanged).
    """
    if not isinstance(updates, dict):
        return None
    out: NodeGradients = {}
    for nid, val in updates.items():
        if isinstance(val, dict):
            out[nid] = {k: np.asarray(_to_numpy(v)) for k, v in val.items()}
        else:
            arr = _to_numpy(val)
            if arr is None:
                return None
            out[nid] = {"_flat": np.asarray(arr)}
    return out


def _to_numpy(x: Any) -> np.ndarray | None:
    """Convert a torch tensor or numpy array to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    if callable(detach):
        try:
            return x.detach().cpu().numpy()
        except Exception:  # pragma: no cover - defensive
            return None
    try:
        return np.asarray(x)
    except Exception:  # pragma: no cover - defensive
        return None
