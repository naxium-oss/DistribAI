"""
Optimizer factory and AuON implementation for DistribAI worker jobs.

This module exposes a small registry-based factory (`build_optimizer`)
that the executor and the DiLoCo runner consume to construct optimizers
by name. AuON is the system default as of v1.2 (authorised by project
owner based on FANT2 internal benchmark, 1.89x wall-time speed-up over
Muon at matched-step loss on Crowfeather-50m).

References
----------
- AuON: Anderson-Updated Optimization Network. arXiv:2509.24320.
- FANT2 internal benchmark notes (2026-04-30 paper_followups).

The implementation here is reconstructed from the paper plus our own
benchmark notes -- there is no upstream reference implementation
published as of 2026-05.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
from torch.optim import SGD, Adam, AdamW

logger = logging.getLogger(__name__)


# Registry of name -> optimizer class.
#
# The registry is intentionally module-level so a sibling agent that
# imports `worker.src.daemon.optimizers.build_optimizer` (for example
# the DiLoCo inner-optimizer factory) can resolve names without
# coupling to this module's internal layout.
_REGISTRY: dict[str, type[torch.optim.Optimizer]] = {}


def register(name: str):
    """Decorator: register an optimizer class under a case-insensitive name."""

    def decorator(cls: type[torch.optim.Optimizer]) -> type[torch.optim.Optimizer]:
        _REGISTRY[name.lower()] = cls
        return cls

    return decorator


def list_registered() -> list[str]:
    """Return the sorted list of registered optimizer names."""

    return sorted(_REGISTRY.keys())


def build_optimizer(
    name: str,
    params: Iterable[torch.nn.Parameter],
    **kwargs,
) -> torch.optim.Optimizer:
    """Build an optimizer by name. Unknown names fall back to AuON.

    Args:
        name: Case-insensitive optimizer name. Must be one of
            ``list_registered()``. If not registered, the factory
            logs a warning and returns an AuON instance.
        params: Parameter iterable, as accepted by torch optimizers.
        **kwargs: Forwarded to the optimizer constructor.

    Returns:
        Instantiated optimizer.
    """

    key = name.lower() if isinstance(name, str) else "auon"
    cls = _REGISTRY.get(key)
    if cls is None:
        logger.warning(
            "Unknown optimizer %r requested, falling back to AuON (system default).",
            name,
        )
        cls = AuON
    return cls(params, **kwargs)


# --------------------------------------------------------------------------- #
# Thin wrappers so the standard torch optimizers participate in the registry.
# --------------------------------------------------------------------------- #


@register("adamw")
class _AdamWWrapper(AdamW):
    """``torch.optim.AdamW`` re-exported under the registry name ``adamw``."""


@register("adam")
class _AdamWrapper(Adam):
    """``torch.optim.Adam`` re-exported under the registry name ``adam``."""


@register("sgd")
class _SGDWrapper(SGD):
    """``torch.optim.SGD`` re-exported under the registry name ``sgd``."""


# --------------------------------------------------------------------------- #
# AuON
# --------------------------------------------------------------------------- #


@register("auon")
class AuON(torch.optim.Optimizer):
    """Anderson-Updated Optimization Network (arxiv:2509.24320).

    Drop-in replacement for Muon/Adam-class optimizers. On the FANT2
    Crowfeather-50m benchmark AuON beat Muon 1.89x in wall time
    (110.9 vs 209.6 ms/step) and produced strictly lower loss across
    500 matched optimisation steps from ``step_017500.pt``. The cost
    advantage comes from an O(MN) cosh-brake update versus Muon's
    O(MN.min(M,N)) Newton-Schulz orthogonalisation.

    System-default optimiser for DistribAI jobs as of v1.2.

    Notes
    -----
    Implementation reconstructed from arxiv:2509.24320 plus FANT2
    internal benchmark notes. No upstream reference implementation
    is published as of 2026-05; values that are "from the paper" are
    flagged inline, anything else is a reasonable engineering choice
    consistent with the paper's described behaviour.

    The core update is:

        # Anderson direction
        d_k = sum_i w_i * g_{k-i}                       (least-squares mix)

        # Cosh brake (paper, Section 3, Equation 7)
        scale = 1 / cosh( ||d_k||_2 / cosh_brake )
        d_k <- scale * d_k

        # AdamW-style decoupled weight decay
        theta_{k+1} = theta_k - lr * (d_k + weight_decay * theta_k)

    The cosh brake prevents the rare runaway-step that pure Anderson
    occasionally produces when the window-residual problem is
    ill-conditioned; the paper's recommended default ``cosh_brake=1.0``
    is used here.

    The Anderson coefficients ``w_i`` solve the small constrained
    least-squares problem ``min_w || sum_i w_i * g_i ||`` subject to
    ``sum_i w_i = 1`` (paper, Section 2.2). The closed-form solution is
    ``w = (G^{-1} 1) / (1^T G^{-1} 1)`` where ``G_ij = g_i . g_j``. At
    window size 1 this degenerates to plain SGD with momentum-free
    updates, matching the paper's stated boundary behaviour.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cosh_brake: float = 1.0,
        anderson_window: int = 4,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"AuON lr must be positive, got {lr}")
        if cosh_brake <= 0.0:
            raise ValueError(f"AuON cosh_brake must be positive, got {cosh_brake}")
        if anderson_window < 1:
            raise ValueError(f"AuON anderson_window must be >= 1, got {anderson_window}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"AuON betas must lie in [0, 1), got {betas}")
        if eps <= 0.0:
            raise ValueError(f"AuON eps must be positive, got {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"AuON weight_decay must be non-negative, got {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "cosh_brake": cosh_brake,
            "anderson_window": anderson_window,
        }
        super().__init__(params, defaults)

    # ----- Anderson mix --------------------------------------------------- #

    @staticmethod
    def _anderson_direction(
        history: list[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        """Return the Anderson-mixed direction given a gradient history.

        ``history[-1]`` is the newest gradient. The returned tensor has
        the same shape and dtype as ``history[-1]``.

        With history of length 1 this is plain SGD. With length k>=2 we
        solve the small constrained least-squares problem

            min_w || sum_i w_i * g_i ||_2     s.t.  sum_i w_i = 1

        where the ``g_i`` are the past gradients (residuals of the
        gradient-descent fixed-point iteration). See arxiv:2509.24320
        Section 2.2. The closed-form solution is

            w = (G^{-1} 1) / (1^T G^{-1} 1),   G_ij = g_i . g_j.

        When the gradient sequence is constant the mix returns that
        gradient unchanged. When two recent residuals are nearly
        collinear we add an ``eps``-scaled ridge so the solve stays
        well-conditioned.
        """

        if len(history) == 1:
            return history[0].clone()

        # Flatten everything for the small least-squares problem.
        flat = [h.reshape(-1) for h in history]
        stacked = torch.stack(flat, dim=0)  # (k, N)
        # Gram matrix of the residuals: G_ij = g_i . g_j.
        gram = stacked @ stacked.T  # (k, k)
        # Ridge for numerical stability.
        gram = gram + eps * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs = torch.ones(gram.shape[0], device=gram.device, dtype=gram.dtype)
        try:
            sol = torch.linalg.solve(gram, rhs)
        except RuntimeError:
            sol = torch.full_like(rhs, 1.0 / gram.shape[0])
        denom = sol.sum().clamp_min(eps)
        w = sol / denom

        mixed = (w.unsqueeze(1) * stacked).sum(dim=0)
        return mixed.reshape(history[-1].shape)

    # ----- step ----------------------------------------------------------- #

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            cosh_brake = group["cosh_brake"]
            window = group["anderson_window"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AuON does not support sparse gradients.")

                state = self.state[p]
                if "history" not in state:
                    state["history"] = []
                    state["step"] = 0

                hist: list[torch.Tensor] = state["history"]
                hist.append(grad.detach().clone())
                if len(hist) > window:
                    del hist[0]
                state["step"] += 1

                direction = self._anderson_direction(hist, eps=eps)

                # Cosh brake. Use a single global L2 norm of the
                # direction (paper, Section 3): this is what gives the
                # O(MN) cost guarantee vs Muon's O(MN min(M,N)).
                norm = direction.norm()
                # cosh grows exponentially, so very large norms give
                # scale -> 0 and the step is effectively suppressed.
                brake = torch.cosh(norm / cosh_brake)
                direction = direction / brake.clamp_min(1.0)

                if wd != 0.0:
                    direction = direction + wd * p.data

                p.data.add_(direction, alpha=-lr)

        return loss


__all__ = [
    "AuON",
    "build_optimizer",
    "list_registered",
    "register",
]
