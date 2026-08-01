"""Unit tests for FoolsGold sybil-similarity defense (arxiv:1808.04866)."""

from __future__ import annotations

import numpy as np

from worker.src.daemon.byzantine_detector.foolsgold import (
    FoolsGold,
    FoolsGoldFilter,
    _flatten_node,
    _stack_nodes,
)


def _honest_node(rng: np.random.Generator, base: np.ndarray, noise: float = 1.0):
    """Honest worker gradient: small shared direction + meaningful local noise.

    Splits ``base`` into a leading weight block (sqrt(d)*sqrt(d) when possible,
    else 1-D) and a trailing bias block of length 4. Caller must supply
    ``base`` of length divisible by suitable factors for the chosen layers.
    """
    d = base.shape[0]
    # Use a flat 1-D weight layer to avoid shape-aware reshape gymnastics.
    weight = 0.3 * base + noise * rng.standard_normal(d)
    bias = 0.1 * base[: min(4, d)] + noise * rng.standard_normal(min(4, d))
    return {"layer1.weight": weight, "layer1.bias": bias}


def test_flatten_node_orders_keys():
    g = {"b": np.array([3.0, 4.0]), "a": np.array([1.0, 2.0])}
    out = _flatten_node(g)
    # Sorted keys -> a then b.
    assert out.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_stack_nodes_handles_ragged_with_warning(caplog):
    g = {
        "n0": {"x": np.ones(5)},
        "n1": {"x": np.ones(7)},
    }
    with caplog.at_level("WARNING"):
        ids, mat = _stack_nodes(g)
    assert mat.shape == (2, 7)
    assert any("ragged" in r.message for r in caplog.records)


def test_compute_weights_two_nodes_honest():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(16)
    grads = {
        "n0": {"w": base + 0.05 * rng.standard_normal(16)},
        "n1": {"w": base + 0.05 * rng.standard_normal(16)},
    }
    fg = FoolsGold(use_logit=False, pardoning=False)
    w = fg.compute_weights(grads)
    # Two nodes pointing same direction -> high similarity -> low linear weight
    assert 0.0 <= w["n0"] <= 1.0
    assert 0.0 <= w["n1"] <= 1.0


def test_honest_baseline_weights_kept_high_no_pardoning():
    """Linear FoolsGold without pardoning on honest workers -> weights ~ uniform."""
    rng = np.random.default_rng(42)
    base = rng.standard_normal(16)
    grads = {f"h{i}": _honest_node(rng, base, noise=1.0) for i in range(6)}
    fg = FoolsGold(use_logit=False, pardoning=False)
    weights = fg.compute_weights(grads)
    assert set(weights.keys()) == set(grads.keys())
    vals = list(weights.values())
    # Without pardoning + meaningful per-node noise, weights cluster well
    # above zero and are roughly uniform.
    assert min(vals) > 0.3, f"honest no-pardon weights should be high: {vals}"
    assert max(vals) - min(vals) < 0.4


def test_logit_on_all_honest_documents_known_pathology():
    """Documented FoolsGold behaviour: with pardoning, all-honest clusters get
    crushed weights because the most-similar honest pair anchors the
    normalisation. Users should set ``pardoning=False`` or ``use_logit=False``
    when no sybils are suspected."""
    rng = np.random.default_rng(42)
    base = rng.standard_normal(16)
    grads = {f"h{i}": _honest_node(rng, base, noise=0.5) for i in range(6)}
    fg = FoolsGold(use_logit=True, pardoning=True)
    weights = fg.compute_weights(grads)
    # With logit + pardoning on all-honest, weights collapse near 0 --
    # this is documented behaviour, not a bug.
    assert max(weights.values()) < 0.6


def test_sybil_attack_penalised():
    """Three identical attackers should get weights << honest nodes."""
    rng = np.random.default_rng(7)
    base = rng.standard_normal(20)
    honest = {f"h{i}": _honest_node(rng, base, noise=0.5) for i in range(6)}
    # Three identical sybil gradients pointing in a different direction.
    sybil_vec = -2.0 * base
    sybils = {
        f"s{i}": {
            "layer1.weight": sybil_vec.copy(),
            "layer1.bias": sybil_vec[:4].copy(),
        }
        for i in range(3)
    }
    grads = {**honest, **sybils}
    fg = FoolsGold(use_logit=False, pardoning=True)
    w = fg.compute_weights(grads)
    sybil_w = np.mean([w[k] for k in sybils])
    honest_w = np.mean([w[k] for k in honest])
    assert sybil_w < honest_w, (
        f"sybils should be down-weighted; sybil={sybil_w:.3f} honest={honest_w:.3f}"
    )
    # Sybils should be heavily penalised (near zero).
    assert sybil_w < 0.1


def test_logit_transform_keeps_weights_in_unit_interval():
    rng = np.random.default_rng(11)
    base = rng.standard_normal(8)
    grads = {f"n{i}": {"w": base + 0.2 * rng.standard_normal(8)} for i in range(5)}
    fg = FoolsGold(use_logit=True)
    w = fg.compute_weights(grads)
    for v in w.values():
        assert 0.0 <= v <= 1.0


def test_single_node_returns_unit_weight():
    fg = FoolsGold()
    w = fg.compute_weights({"only": {"x": np.array([1.0, 2.0, 3.0])}})
    assert w == {"only": 1.0}


def test_filter_noop_below_min_nodes():
    fg = FoolsGoldFilter(min_nodes=4)
    grads = {f"n{i}": {"x": np.ones(3)} for i in range(3)}
    out = fg(grads)
    # Pass-through when fewer than 4 nodes.
    assert out is grads
    assert fg.last_weights == {f"n{i}": 1.0 for i in range(3)}


def test_filter_applies_weights_when_enough_nodes():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(8)
    honest = {f"h{i}": {"w": base + 0.5 * rng.standard_normal(8)} for i in range(4)}
    sybils = {f"s{i}": {"w": -base.copy()} for i in range(3)}
    fg = FoolsGoldFilter(min_nodes=4, use_logit=False)
    out = fg({**honest, **sybils})
    # The sybils' arrays should be scaled down vs originals (heavily).
    for sid in sybils:
        before_norm = float(np.linalg.norm(sybils[sid]["w"]))
        after_norm = float(np.linalg.norm(out[sid]["w"]))
        assert after_norm <= before_norm + 1e-8


def test_numerical_stability_small_gradients():
    """Cosine math must not blow up when all gradients are near zero."""
    grads = {f"n{i}": {"w": np.full(10, 1e-12)} for i in range(5)}
    fg = FoolsGold(epsilon=1e-8)
    # Should not raise / nan-out.
    w = fg.compute_weights(grads)
    for v in w.values():
        assert np.isfinite(v)
        assert 0.0 <= v <= 1.0
