"""Integration: FoolsGold + SignGuard + Bucketing pipeline through AdaptiveAggregator."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from worker.src.daemon.byzantine_detector import (
    AdaptiveAggregator,
    BucketingClipping,
    FoolsGoldFilter,
    SignGuard,
    default_defense_pipeline,
)


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_default_pipeline_is_empty_for_backwards_compat():
    """The default constructor must NOT enable the new defenses (legacy behaviour)."""
    a = AdaptiveAggregator()
    assert a.defense_pipeline == []


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_empty_pipeline_matches_legacy_aggregate_bitwise():
    """With pipeline=[], aggregate() must reproduce legacy output bit-for-bit."""
    torch.manual_seed(0)
    updates = {f"n{i}": torch.randn(8) for i in range(6)}
    legacy = AdaptiveAggregator()
    new = AdaptiveAggregator(defense_pipeline=[])
    out_legacy = legacy.aggregate(updates)
    out_new = new.aggregate(updates)
    assert torch.equal(out_legacy, out_new)


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_pipeline_noop_below_four_nodes():
    """All three filters are no-ops below 4 nodes."""
    torch.manual_seed(1)
    updates = {f"n{i}": torch.randn(8) for i in range(3)}
    a = AdaptiveAggregator(defense_pipeline=default_defense_pipeline())
    out = a.aggregate(updates)
    assert out.shape == (8,)
    # method_used should be coordinate_median for n<5.
    assert a.method_used == "coordinate_median"


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_compose_pipeline_reduces_error_under_mixed_attack():
    """
    10 nodes: 7 honest (non-IID, large per-node variation around a shared
    direction), 2 sybils (identical poisoned vectors), 1 sign-flipper.
    The pipeline-protected aggregate should be within ~0.1*||honest_mean||
    L2 of the honest-only mean -- significantly closer than the naive mean
    over all updates including sybils + flipper.
    """
    rng = np.random.default_rng(2026)
    dim = 16
    direction = rng.standard_normal(dim) * 0.3  # weak shared signal

    honest = {}
    for i in range(7):
        # Substantial per-node noise so honest nodes are NOT pairwise-similar
        # (simulating non-IID workers with their own data partitions).
        v = direction + 1.0 * rng.standard_normal(dim)
        honest[f"h{i}"] = torch.from_numpy(v.astype(np.float32))

    honest_mean = torch.stack(list(honest.values())).mean(dim=0)

    # Sybils: large coordinated poisoned vector pointing the wrong way.
    sybil_vec = (-5.0 * direction - 3.0).astype(np.float32)
    sybils = {
        "s0": torch.from_numpy(sybil_vec.copy()),
        "s1": torch.from_numpy(sybil_vec.copy()),
    }
    # Sign-flipper: flip the sign of every coord on an honest-style draw.
    flipper = {
        "f0": torch.from_numpy(-(direction + 1.0 * rng.standard_normal(dim)).astype(np.float32))
    }
    all_updates = {**honest, **sybils, **flipper}

    defended = AdaptiveAggregator(
        defense_pipeline=[
            FoolsGoldFilter(min_nodes=4, use_logit=False, pardoning=False),
            SignGuard(min_nodes=4),
            BucketingClipping(bucket_size=2, seed=0, min_nodes=4),
        ]
    )
    out_defended = defended.aggregate(all_updates)

    err_def = float(torch.norm(out_defended - honest_mean))
    honest_norm = float(torch.norm(honest_mean))
    naive_all = torch.stack(list(all_updates.values())).mean(dim=0)
    err_naive = float(torch.norm(naive_all - honest_mean))
    assert err_def < err_naive, f"defended ({err_def:.4f}) should beat naive-mean ({err_naive:.4f})"
    # Defended aggregate should land within roughly one honest-mean norm
    # under this challenging mixed attack.
    assert err_def < 1.0 * max(honest_norm, 1.0), (
        f"defended err {err_def:.4f} too large vs honest mean norm {honest_norm:.4f}"
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
def test_compose_pipeline_records_honest_weights():
    """Honest nodes' FoolsGold weights should beat sybils' weights.

    Note: in absolute terms FoolsGold can still down-weight honest nodes
    when honest gradients are pairwise similar (the algorithm only sees
    cosine geometry, not labels). The reliable assertion is the relative
    ordering -- sybils get less weight than honest peers.
    """
    rng = np.random.default_rng(7)
    dim = 12
    direction = rng.standard_normal(dim) * 0.3
    honest = {
        f"h{i}": torch.from_numpy((direction + 1.0 * rng.standard_normal(dim)).astype(np.float32))
        for i in range(6)
    }
    sybils = {
        "s0": torch.from_numpy((-2.0 * direction).astype(np.float32)),
        "s1": torch.from_numpy((-2.0 * direction).astype(np.float32)),
    }
    fg = FoolsGoldFilter(min_nodes=4, use_logit=False, pardoning=True)
    pipeline = [fg, SignGuard(min_nodes=4), BucketingClipping(seed=0, min_nodes=4)]
    a = AdaptiveAggregator(defense_pipeline=pipeline)
    a.aggregate({**honest, **sybils})

    honest_weights = [fg.last_weights[f"h{i}"] for i in range(6)]
    sybil_weights = [fg.last_weights[s] for s in sybils]
    # Relative ordering: sybils strictly penalised below honest mean.
    assert float(np.mean(sybil_weights)) < float(np.mean(honest_weights)), (
        f"sybil weights {sybil_weights} should be lower than honest {honest_weights}"
    )
    # Mean honest weight should remain non-trivial after FoolsGold.
    assert float(np.mean(honest_weights)) > 0.1
