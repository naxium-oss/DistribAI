"""Unit tests for Bucketing + Centered Clipping (arxiv:2202.01545)."""

from __future__ import annotations

import numpy as np
import pytest

from worker.src.daemon.byzantine_detector.bucketing import BucketingClipping


def _make_grads(n: int, dim: int = 8) -> dict:
    return {f"n{i}": {"w": np.full(dim, float(i), dtype=np.float64)} for i in range(n)}


def test_bucket_count_n10_s2():
    bc = BucketingClipping(bucket_size=2, seed=0)
    grads = _make_grads(10)
    buckets = bc.bucket(grads)
    assert len(buckets) == 5


def test_bucket_count_n11_s2_has_leftover():
    bc = BucketingClipping(bucket_size=2, seed=0)
    grads = _make_grads(11)
    buckets = bc.bucket(grads)
    # ceil(11/2) = 6
    assert len(buckets) == 6


def test_each_bucket_is_mean_of_its_members():
    """n=10, s=2 -> 5 buckets, each output = mean of its bucket members."""
    bc = BucketingClipping(bucket_size=2, seed=7)
    grads = _make_grads(10, dim=4)
    # Re-derive the permutation with the same seed by inspecting bucket means.
    rng = np.random.default_rng(7)
    perm = rng.permutation(10)
    buckets = bc.bucket(grads)
    assert len(buckets) == 5
    for i, b in enumerate(buckets):
        idx = perm[2 * i : 2 * i + 2]
        expected = np.mean([float(j) for j in idx])
        np.testing.assert_allclose(b, np.full(4, expected))


def test_bucket_dict_preserves_layer_structure():
    bc = BucketingClipping(bucket_size=2, seed=42)
    grads = {
        f"n{i}": {
            "layer1": np.full(4, float(i)),
            "layer2": np.full((2, 3), float(i)),
        }
        for i in range(6)
    }
    out = bc.bucket_dict(grads)
    assert len(out) == 3
    for nid, layers in out.items():
        assert nid.startswith("bucket_")
        assert set(layers.keys()) == {"layer1", "layer2"}
        assert layers["layer1"].shape == (4,)
        assert layers["layer2"].shape == (2, 3)


def test_clip_inside_radius_unchanged():
    bc = BucketingClipping(tau=10.0)
    centre = np.zeros(4)
    updates = [np.array([1.0, 1.0, 1.0, 1.0])]  # norm = 2
    out = bc.clip(updates, centre, tau=10.0)
    np.testing.assert_allclose(out[0], updates[0])


def test_clip_outside_radius_projected():
    bc = BucketingClipping()
    centre = np.zeros(4)
    u = np.array([10.0, 0.0, 0.0, 0.0])
    out = bc.clip([u], centre, tau=3.0)
    # Projected onto tau-ball: direction preserved, magnitude == tau.
    np.testing.assert_allclose(np.linalg.norm(out[0] - centre), 3.0, atol=1e-6)
    # Direction preserved.
    np.testing.assert_allclose(out[0] / np.linalg.norm(out[0]), u / np.linalg.norm(u))


def test_clip_zero_tau_rejected():
    bc = BucketingClipping()
    with pytest.raises(ValueError):
        bc.clip([np.zeros(3)], np.zeros(3), tau=0.0)


def test_update_centre_first_call_initialises():
    bc = BucketingClipping(momentum=0.9)
    v = np.array([1.0, 2.0, 3.0])
    out = bc.update_centre(v)
    np.testing.assert_allclose(out, v)
    np.testing.assert_allclose(bc.centre, v)


def test_update_centre_ema():
    bc = BucketingClipping(momentum=0.5)
    bc.update_centre(np.array([0.0, 0.0]))
    out = bc.update_centre(np.array([4.0, 8.0]))
    # 0.5 * 0 + 0.5 * 4 = 2 ; 0.5 * 0 + 0.5 * 8 = 4
    np.testing.assert_allclose(out, [2.0, 4.0])


def test_call_below_min_nodes_noop():
    bc = BucketingClipping(bucket_size=2, min_nodes=4, seed=0)
    grads = _make_grads(3, dim=4)
    out = bc(grads)
    assert out is grads  # pass-through


def test_call_above_min_nodes_buckets():
    bc = BucketingClipping(bucket_size=2, min_nodes=4, seed=0)
    grads = _make_grads(8, dim=4)
    out = bc(grads)
    assert len(out) == 4
    for nid in out:
        assert nid.startswith("bucket_")
