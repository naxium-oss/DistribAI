"""Unit tests for SignGuard sign-statistics defense (arxiv:2109.05872)."""

from __future__ import annotations

import numpy as np

from worker.src.daemon.byzantine_detector.signguard import SignGuard


def _make_honest(rng: np.random.Generator, n: int, dim: int = 32) -> dict:
    """Create n honest nodes with gradients tightly clustered around a common direction."""
    direction = rng.standard_normal(dim)
    grads = {}
    for i in range(n):
        # Low noise so honest nodes agree on the sign in the majority of dims.
        grads[f"h{i}"] = {"w": direction + 0.05 * rng.standard_normal(dim)}
    return grads, direction


def test_honest_baseline_kept():
    rng = np.random.default_rng(0)
    grads, _ = _make_honest(rng, n=10, dim=32)
    sg = SignGuard()
    kept, flagged = sg.filter(grads)
    # Tight cluster -> no statistically low outliers -> all kept.
    assert flagged == set()
    assert set(kept.keys()) == set(grads.keys())


def test_sign_flip_attack_flags_exactly_two():
    """2 of 10 nodes flip the sign of every coordinate -> SignGuard flags those 2."""
    rng = np.random.default_rng(123)
    grads, direction = _make_honest(rng, n=8, dim=64)
    # Two sign-flipped attackers.
    grads["bad0"] = {"w": -(direction + 0.05 * rng.standard_normal(64))}
    grads["bad1"] = {"w": -(direction + 0.05 * rng.standard_normal(64))}

    sg = SignGuard()
    kept, flagged = sg.filter(grads)
    assert flagged == {"bad0", "bad1"}, f"got flagged={flagged}"
    assert set(kept.keys()) == {f"h{i}" for i in range(8)}


def test_noop_below_min_nodes():
    sg = SignGuard(min_nodes=4)
    grads = {f"n{i}": {"w": np.array([1.0, -1.0])} for i in range(3)}
    kept, flagged = sg.filter(grads)
    assert kept == grads
    assert flagged == set()


def test_never_flags_all_nodes():
    """Degenerate input where every node has identical agreement -> no flags."""
    rng = np.random.default_rng(7)
    g = rng.standard_normal(20)
    grads = {f"n{i}": {"w": g.copy()} for i in range(5)}
    sg = SignGuard(z_threshold=0.5)
    kept, flagged = sg.filter(grads)
    assert flagged == set()
    assert len(kept) == 5


def test_agreement_scores_in_zero_one():
    rng = np.random.default_rng(2)
    grads, _ = _make_honest(rng, n=6, dim=40)
    sg = SignGuard()
    sg.filter(grads)
    for v in sg.last_scores.values():
        assert 0.0 <= v <= 1.0


def test_filter_returns_independent_kept_dict():
    rng = np.random.default_rng(99)
    grads, direction = _make_honest(rng, n=8, dim=32)
    # Two sign-flippers needed since one outlier inflates MAD of its own group.
    grads["bad0"] = {"w": -(direction + 0.05 * rng.standard_normal(32))}
    grads["bad1"] = {"w": -(direction + 0.05 * rng.standard_normal(32))}
    sg = SignGuard()
    kept, flagged = sg.filter(grads)
    # Modifying kept must not modify the original.
    assert "bad0" not in kept and "bad1" not in kept
    # Ndarray references are shared (no copy) -- this is intentional and safe
    # because downstream aggregators do not mutate inputs.
    nid = next(iter(kept))
    assert kept[nid]["w"] is grads[nid]["w"]


def test_call_returns_only_kept():
    rng = np.random.default_rng(31)
    grads, direction = _make_honest(rng, n=8, dim=32)
    grads["bad0"] = {"w": -(direction + 0.05 * rng.standard_normal(32))}
    grads["bad1"] = {"w": -(direction + 0.05 * rng.standard_normal(32))}
    sg = SignGuard()
    out = sg(grads)
    assert "bad0" not in out and "bad1" not in out
    assert len(out) == 8
