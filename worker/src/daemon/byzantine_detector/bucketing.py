"""Bucketing + centered clipping defense for non-IID federated learning."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from .foolsgold import NodeGradients, _flatten_node

logger = logging.getLogger(__name__)


class BucketingClipping:
    """Bucketing + centered clipping for heterogeneous data (arxiv:2202.01545).

    Implements the bucketing pre-processing step and the centered-clipping
    post-processing step of He, Karimireddy and Jaggi (ICLR 2022). The two
    pieces are designed to wrap an existing Byzantine-robust aggregator
    (Krum, Multi-Krum, trimmed-mean, ...) so that it remains correct under
    non-IID worker data — a regime where vanilla Krum is known to fail
    silently because honest workers look "Byzantine" relative to each other.

    The bucketing step randomly partitions ``n`` node updates into
    ``ceil(n/s)`` size-``s`` buckets and averages within each bucket. The
    smoothed updates are then passed to the base aggregator. Centered
    clipping projects each update onto the L2 ball of radius ``tau`` around
    a momentum-tracked centre, capping the per-step influence of any single
    update.

    Pure-NumPy implementation operating on
    ``dict[str, dict[str, np.ndarray]]`` (node_id -> layer_name -> ndarray).

    Attributes:
        bucket_size: Default ``s`` (paper uses 2 for honest-majority
            settings; larger ``s`` gives more smoothing).
        tau: Default clipping radius for :meth:`clip`.
        momentum: Momentum coefficient for the running centre used by
            :meth:`update_centre`.
        rng: NumPy ``Generator`` used to draw the random partition. Pass a
            seeded generator for reproducible tests.
        centre: Most-recent momentum-tracked centre vector (set lazily on
            the first call to :meth:`update_centre`).
    """

    def __init__(
        self,
        bucket_size: int = 2,
        tau: float = 10.0,
        momentum: float = 0.9,
        seed: int | None = None,
        min_nodes: int = 4,
    ) -> None:
        self.bucket_size = max(1, int(bucket_size))
        self.tau = float(tau)
        self.momentum = float(momentum)
        self.min_nodes = int(min_nodes)
        self.rng = np.random.default_rng(seed)
        self.centre: np.ndarray | None = None

    # ------------------------------------------------------------------ bucket
    def bucket(
        self,
        node_gradients: NodeGradients,
        bucket_size: int | None = None,
    ) -> list[np.ndarray]:
        """Random-partition nodes into buckets and average within each bucket.

        Args:
            node_gradients: Mapping of node_id to per-layer gradient ndarrays.
            bucket_size: Override the instance-default ``bucket_size``.

        Returns:
            List of flattened bucket-averaged gradient vectors, one per
            bucket. Length is ``ceil(n / s)`` where ``n = len(node_gradients)``
            and ``s = bucket_size``. The last bucket may have fewer than
            ``s`` members if ``n`` is not divisible by ``s``.
        """
        s = max(1, int(bucket_size if bucket_size is not None else self.bucket_size))
        ids = list(node_gradients.keys())
        n = len(ids)
        if n == 0:
            return []
        vecs = np.stack([_flatten_node(node_gradients[nid]) for nid in ids], axis=0)
        perm = self.rng.permutation(n)
        buckets: list[np.ndarray] = []
        for start in range(0, n, s):
            idx = perm[start : start + s]
            buckets.append(vecs[idx].mean(axis=0))
        return buckets

    def bucket_dict(
        self,
        node_gradients: NodeGradients,
        bucket_size: int | None = None,
    ) -> NodeGradients:
        """Bucket and re-pack as a NodeGradients dict with synthetic ids.

        Args:
            node_gradients: Mapping of node_id to per-layer gradient ndarrays.
            bucket_size: Override the instance-default ``bucket_size``.

        Returns:
            A new NodeGradients dict with one entry per bucket, keyed
            ``"bucket_<i>"``. Layer structure of the original gradients is
            preserved: each layer is averaged across the bucket's members.
        """
        s = max(1, int(bucket_size if bucket_size is not None else self.bucket_size))
        ids = list(node_gradients.keys())
        n = len(ids)
        if n == 0:
            return {}
        layer_keys = sorted(node_gradients[ids[0]].keys()) if ids else []
        perm = self.rng.permutation(n)
        out: NodeGradients = {}
        bucket_idx = 0
        for start in range(0, n, s):
            members = [ids[j] for j in perm[start : start + s]]
            averaged: dict[str, np.ndarray] = {}
            for k in layer_keys:
                stack = np.stack([np.asarray(node_gradients[m][k]) for m in members], axis=0)
                averaged[k] = stack.mean(axis=0)
            out[f"bucket_{bucket_idx}"] = averaged
            bucket_idx += 1
        return out

    # ------------------------------------------------------------------- clip
    def clip(
        self,
        updates: Sequence[np.ndarray],
        centre: np.ndarray,
        tau: float | None = None,
    ) -> list[np.ndarray]:
        """Project each update onto the L2 ball of radius ``tau`` around ``centre``.

        Args:
            updates: Iterable of 1-D update vectors (typically the bucket
                outputs from :meth:`bucket`).
            centre: The 1-D centre vector to clip around. Must have the same
                length as each update.
            tau: Override the instance-default clipping radius.

        Returns:
            List of clipped update vectors. Each output equals
            ``centre + min(1, tau / ||u - centre||) * (u - centre)``.
        """
        radius = float(tau if tau is not None else self.tau)
        if radius <= 0:
            raise ValueError(f"tau must be positive, got {radius}")
        clipped: list[np.ndarray] = []
        for u in updates:
            diff = u - centre
            norm = float(np.linalg.norm(diff))
            scale = 1.0 if norm <= radius else radius / (norm + 1e-12)
            clipped.append(centre + scale * diff)
        return clipped

    def update_centre(self, new_estimate: np.ndarray) -> np.ndarray:
        """EMA-update the momentum-tracked centre and return the new value.

        Args:
            new_estimate: Latest aggregated update vector.

        Returns:
            Updated centre = ``momentum * centre + (1 - momentum) * estimate``.
            On the first call the centre is initialised to ``new_estimate``.
        """
        if self.centre is None or self.centre.shape != new_estimate.shape:
            self.centre = new_estimate.copy()
        else:
            self.centre = self.momentum * self.centre + (1.0 - self.momentum) * new_estimate
        return self.centre

    # ------------------------------------------------------------------- call
    def __call__(self, node_gradients: NodeGradients) -> NodeGradients:
        """Bucket the gradients for the pipeline (no clip; clip needs a centre).

        Below ``min_nodes`` this is a no-op so small clusters are unaffected.
        """
        if len(node_gradients) < self.min_nodes:
            logger.debug(
                "BucketingClipping: %d nodes < min_nodes=%d, skipping",
                len(node_gradients),
                self.min_nodes,
            )
            return node_gradients
        bucketed = self.bucket_dict(node_gradients)
        logger.debug(
            "BucketingClipping: %d nodes -> %d buckets (s=%d)",
            len(node_gradients),
            len(bucketed),
            self.bucket_size,
        )
        return bucketed
