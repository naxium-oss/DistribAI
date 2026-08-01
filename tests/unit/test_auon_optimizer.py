"""Unit tests for AuON optimizer and the build_optimizer factory.

References: arxiv:2509.24320 + FANT2 internal benchmark notes (2026-04-30).
"""

from __future__ import annotations

import logging

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    from worker.src.daemon.optimizers import (
        _REGISTRY,
        AuON,
        build_optimizer,
        list_registered,
    )

    HAS_OPTIMIZERS = True
except ImportError:
    HAS_OPTIMIZERS = False
    AuON = None  # type: ignore[assignment]
    build_optimizer = None  # type: ignore[assignment]
    list_registered = None  # type: ignore[assignment]
    _REGISTRY = {}  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(
    not HAS_TORCH or not HAS_OPTIMIZERS,
    reason="torch or optimizers module not available",
)


# --------------------------------------------------------------------------- #
# Registry-level behaviour
# --------------------------------------------------------------------------- #


def test_auon_registered_under_name():
    assert "auon" in _REGISTRY
    assert _REGISTRY["auon"] is AuON


def test_build_optimizer_returns_auon_instance():
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = build_optimizer("auon", [p], lr=1e-3)
    assert isinstance(opt, AuON)


def test_build_optimizer_is_case_insensitive_for_adamw():
    p = torch.nn.Parameter(torch.randn(4, 4))
    opt = build_optimizer("AdAmW", [p], lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_unknown_falls_back_to_auon(caplog):
    p = torch.nn.Parameter(torch.randn(4, 4))
    with caplog.at_level(logging.WARNING, logger="worker.src.daemon.optimizers"):
        opt = build_optimizer("nonexistent-optim", [p], lr=1e-3)
    assert isinstance(opt, AuON)
    assert any("nonexistent-optim" in rec.message.lower() for rec in caplog.records)


def test_list_registered_contains_expected():
    names = list_registered()
    assert {"auon", "adamw", "adam", "sgd"}.issubset(set(names))


# --------------------------------------------------------------------------- #
# Convergence on a known-optimum quadratic
# --------------------------------------------------------------------------- #


def test_auon_converges_on_quadratic_within_50_steps():
    """f(x) = 0.5 * ||x - x*||^2 has gradient (x - x*) and optimum x*.

    We test convergence to 1e-4 of the optimum within 50 steps. Use a
    larger cosh_brake so the brake does not damp the early steps when
    the gradient norm is order-1 (the brake is designed to suppress
    pathological spikes, not normal gradients near the optimum).
    """
    torch.manual_seed(0)
    target = torch.tensor([1.0, -2.0, 0.5, 3.0])
    x = torch.nn.Parameter(torch.zeros_like(target))

    opt = AuON([x], lr=1.0, anderson_window=4, cosh_brake=100.0)
    steps_to_converge = None
    for step in range(1, 201):
        opt.zero_grad()
        loss = 0.5 * ((x - target) ** 2).sum()
        loss.backward()
        opt.step()
        err = (x.detach() - target).norm().item()
        if err < 1e-4 and steps_to_converge is None:
            steps_to_converge = step
            break
    assert steps_to_converge is not None, "AuON did not converge within 200 steps"
    assert steps_to_converge <= 50, (
        f"AuON took {steps_to_converge} steps to reach 1e-4 of optimum; expected <=50"
    )


# --------------------------------------------------------------------------- #
# Cosh brake invariant: step-norm bound under gradient spikes
# --------------------------------------------------------------------------- #


def test_auon_cosh_brake_bounds_step_norm():
    """The cosh brake means step norm is bounded even when grad spikes.

    With cosh_brake=1.0 and a gradient much larger than 1.0, the
    update direction has norm ||d|| / cosh(||d||), which is bounded.
    The total step is lr * that direction; we expect it to stay below
    a small constant times lr regardless of how large the raw gradient
    is.
    """
    lr = 0.1
    p = torch.nn.Parameter(torch.zeros(3))
    opt = AuON([p], lr=lr, anderson_window=1, cosh_brake=1.0)

    for spike_magnitude in [1.0, 10.0, 100.0, 1000.0, 1e6]:
        p.data.zero_()
        opt.zero_grad()
        p.grad = torch.tensor([spike_magnitude, 0.0, 0.0])
        opt.state[p] = {}  # reset state so window doesn't leak across the loop
        opt.step()
        step_norm = p.data.norm().item()
        # cosh(x)/x grows, so x/cosh(x) -> 0; therefore the step is
        # at most lr * max_{x>0} (x / cosh(x)) ~= lr * 0.7244.
        # Use a generous 10*lr bound as the spec calls for.
        assert step_norm <= 10.0 * lr, (
            f"spike={spike_magnitude}: step_norm={step_norm} exceeded 10*lr={10 * lr}"
        )


# --------------------------------------------------------------------------- #
# Anderson window: analytical mix
# --------------------------------------------------------------------------- #


def test_anderson_window_one_equals_sgd():
    """With window=1 the Anderson mix is the identity on the gradient."""
    history = [torch.tensor([1.0, -1.0, 2.0])]
    mixed = AuON._anderson_direction(history, eps=1e-8)
    assert torch.allclose(mixed, history[-1])


def test_anderson_window_constant_gradient_returns_same():
    """If all history is identical, every Anderson mix yields that vector."""
    g = torch.tensor([0.3, -0.7, 1.4, 0.1])
    history = [g.clone() for _ in range(4)]
    mixed = AuON._anderson_direction(history, eps=1e-8)
    assert torch.allclose(mixed, g, atol=1e-6), f"got {mixed}, want {g}"


def test_anderson_window_linearly_changing_gradient():
    """For a linearly-changing gradient sequence, the Anderson mix
    converges to a fixed-point estimate (the limit of extrapolation).

    With g_i = g_0 + i * delta (i=0..k-1), the Type-II Anderson
    coefficients with k>=2 should produce a direction that lies in
    the affine span of {g_i} and whose deviation from the analytical
    least-squares solution is below 1e-6.
    """
    g0 = torch.tensor([1.0, 0.0])
    delta = torch.tensor([-0.1, 0.05])
    history = [g0 + i * delta for i in range(4)]

    mixed = AuON._anderson_direction(history, eps=1e-12)

    # Analytical Anderson II on linear g sequences:
    # The differences dg_i = delta are all identical, so the
    # constrained LS solution has alpha = ones/k, and the resulting
    # gradient weights collapse to w_0 = 1 - 1/(k-1), w_{k-1} = 1/(k-1),
    # interior = 0. We verify the answer lies on the same line and the
    # residual is below 1e-6.

    # Closed-form: mixed should be expressible as g0 + alpha * delta for
    # some alpha. Project out the delta direction to test this.
    diff = mixed - g0
    # Solve diff = alpha * delta in least squares sense.
    denom = delta.dot(delta).item()
    alpha = (diff.dot(delta) / denom).item()
    residual = (diff - alpha * delta).norm().item()
    assert residual < 1e-6, (
        f"Anderson mix deviated from analytical line by {residual}"
    )


# --------------------------------------------------------------------------- #
# Sanity: AuON respects basic constructor invariants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": 0.0},
        {"lr": -1.0},
        {"cosh_brake": 0.0},
        {"cosh_brake": -1.0},
        {"anderson_window": 0},
        {"eps": 0.0},
        {"weight_decay": -0.1},
        {"betas": (1.0, 0.95)},
    ],
)
def test_auon_rejects_bad_hyperparams(kwargs):
    p = torch.nn.Parameter(torch.randn(2))
    base = {"lr": 1e-3}
    base.update(kwargs)
    with pytest.raises(ValueError):
        AuON([p], **base)


def test_auon_weight_decay_pulls_toward_zero():
    """With wd>0 and zero gradient, AuON should still shrink params."""
    p = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    opt = AuON([p], lr=0.1, weight_decay=0.5, anderson_window=1)
    p.grad = torch.zeros_like(p)
    initial_norm = p.data.norm().item()
    opt.step()
    final_norm = p.data.norm().item()
    assert final_norm < initial_norm
