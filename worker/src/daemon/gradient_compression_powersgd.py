"""PowerSGD low-rank gradient compression hook (arxiv:1905.13727).

This module provides:

1. ``install_powersgd`` -- a convenience wrapper around PyTorch's
   built-in DDP communication hook for PowerSGD. Use this when the
   model is wrapped in ``torch.nn.parallel.DistributedDataParallel``
   and a process group is initialised.

2. ``PowerSGDCompressor`` -- a standalone, pure-PyTorch implementation
   suitable for the single-node case where DDP is not initialised
   (single-worker development, integration tests, byzantine-detector
   fixtures, etc.). It implements rank-r power iteration with
   warm-started P/Q matrices and error-feedback residual buffers.

Why a separate module from ``gradient_compression.py``?
-------------------------------------------------------
The existing ``gradient_compression.py`` is consumed by the byzantine
detector tests and other call sites that already pin its API. This
module adds the v1.2 PowerSGD wiring without disturbing those callers.

References
----------
- Vogels, Karimireddy, Jaggi. "PowerSGD: Practical Low-Rank Gradient
  Compression for Distributed Optimization". NeurIPS 2019,
  arxiv:1905.13727.
- PyTorch >= 1.10 built-in:
  ``torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# Optional DDP comm-hook import. PowerSGD has shipped with PyTorch
# since 1.10 but we keep the optional gate so that single-node CPU
# installs (no NCCL, no Gloo) don't crash on import.
try:
    from torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook import (
        PowerSGDState,
        powerSGD_hook,
    )

    POWERSGD_AVAILABLE = True
except Exception:  # pragma: no cover -- import-time fallback only
    PowerSGDState = None  # type: ignore[assignment]
    powerSGD_hook = None  # type: ignore[assignment]  # noqa: N816
    POWERSGD_AVAILABLE = False


# --------------------------------------------------------------------------- #
# DDP comm-hook helper.
# --------------------------------------------------------------------------- #


def install_powersgd(
    model: Any,
    *,
    matrix_approximation_rank: int = 4,
    warm_start: bool = True,
    use_error_feedback: bool = True,
    start_step: int = 1000,
    process_group: Any | None = None,
) -> bool:
    """Install PowerSGD as a DDP all-reduce communication hook.

    Args:
        model: A ``torch.nn.parallel.DistributedDataParallel`` instance.
        matrix_approximation_rank: PowerSGD low-rank approximation rank
            (paper recommends 4 for transformer language models).
        warm_start: Reuse the P matrix from the previous iteration as the
            initial guess (paper, Algorithm 1).
        use_error_feedback: Subtract the previous step's reconstruction
            residual from the next compression input (paper, Section 3.3).
        start_step: Number of vanilla all-reduce steps before compression
            kicks in. The paper recommends a grace period to give the
            optimiser a chance to escape the noisy-init regime before
            low-rank approximation is introduced.
        process_group: Optional explicit process group. Default ``None``
            means use the default group.

    Returns:
        ``True`` on success, ``False`` if PowerSGD is unavailable or no
        distributed process group is initialised. Failing soft is
        intentional: the caller should still be able to train on a
        single node without crashing.
    """

    if not POWERSGD_AVAILABLE:
        logger.warning(
            "PowerSGD comm-hook not available -- "
            "torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook not importable."
        )
        return False

    try:
        pg_initialised = dist.is_available() and dist.is_initialized()
    except Exception:
        pg_initialised = False
    if not pg_initialised:
        logger.info(
            "install_powersgd: no distributed process group initialised; "
            "skipping (single-node fallback)."
        )
        return False

    if not hasattr(model, "register_comm_hook"):
        logger.warning(
            "install_powersgd: model has no register_comm_hook; "
            "is it wrapped in DistributedDataParallel?"
        )
        return False

    pg = process_group if process_group is not None else dist.group.WORLD
    state = PowerSGDState(
        process_group=pg,
        matrix_approximation_rank=matrix_approximation_rank,
        warm_start=warm_start,
        use_error_feedback=use_error_feedback,
        start_powerSGD_iter=start_step,
    )
    model.register_comm_hook(state, powerSGD_hook)
    logger.info(
        "PowerSGD comm-hook installed (rank=%d, warm_start=%s, "
        "error_feedback=%s, start_step=%d).",
        matrix_approximation_rank,
        warm_start,
        use_error_feedback,
        start_step,
    )
    return True


# --------------------------------------------------------------------------- #
# Standalone CPU/single-node compressor (no DDP required).
# --------------------------------------------------------------------------- #


def _orthogonalize(matrix: torch.Tensor) -> torch.Tensor:
    """Return a column-orthonormal version of ``matrix`` via QR."""

    if matrix.numel() == 0:
        return matrix
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


class PowerSGDCompressor:
    """Standalone PowerSGD-style low-rank gradient compressor.

    This is a single-process implementation that gives the same
    compression ratio and reconstruction quality as the PyTorch DDP
    comm-hook but does not require a process group. It is useful when
    a worker is running outside DDP -- the single-node case in
    DistribAI is exactly this -- and we want to test or apply PowerSGD
    semantics anyway.

    The state is per-parameter and kept across ``compress`` calls:

    - ``self.p_matrices`` -- warm-started left-projection matrices.
    - ``self.q_matrices`` -- warm-started right-projection matrices.
    - ``self.residual_buffers`` -- error-feedback residuals.

    For a gradient G of shape (d, m) and rank r, compress() returns
    (P, Q) with shapes (d, r) and (r, m). The reconstruction is
    ``P @ Q`` which has shape (d, m); the on-the-wire payload is
    ``d*r + r*m`` numbers instead of ``d*m``, giving the headline
    compression ratio ``d*m / (d*r + r*m)``.
    """

    def __init__(
        self,
        rank: int = 4,
        *,
        warm_start: bool = True,
        use_error_feedback: bool = True,
        num_power_iters: int = 1,
        random_seed: int = 0,
    ) -> None:
        if rank < 1:
            raise ValueError(f"PowerSGD rank must be >= 1, got {rank}")
        if num_power_iters < 1:
            raise ValueError(
                f"PowerSGD num_power_iters must be >= 1, got {num_power_iters}"
            )
        self.rank = rank
        self.warm_start = warm_start
        self.use_error_feedback = use_error_feedback
        self.num_power_iters = num_power_iters
        self._rng = torch.Generator()
        self._rng.manual_seed(random_seed)

        self.p_matrices: dict[str, torch.Tensor] = {}
        self.q_matrices: dict[str, torch.Tensor] = {}
        self.residual_buffers: dict[str, torch.Tensor] = {}

    # ----- public API ---------------------------------------------------- #

    def compress(
        self, gradients: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, Any]]:
        """Compress a name->gradient dictionary, returning P/Q pairs."""

        out: dict[str, dict[str, Any]] = {}
        for name, grad in gradients.items():
            if grad is None or grad.numel() == 0:
                continue
            original_shape = tuple(grad.shape)
            mat = self._as_matrix(grad)
            d, m = mat.shape
            rank = min(self.rank, d, m)

            # Error feedback: add previously-uncompressed residual.
            if self.use_error_feedback and name in self.residual_buffers:
                mat = mat + self.residual_buffers[name]

            # Warm-started Q init; otherwise random.
            need_init = (
                name not in self.q_matrices
                or self.q_matrices[name].shape != (m, rank)
                or self.p_matrices[name].shape != (d, rank)
                or not self.warm_start
            )
            if need_init:
                self.q_matrices[name] = torch.randn(
                    m,
                    rank,
                    generator=self._rng,
                    dtype=mat.dtype,
                    device=mat.device,
                )
                self.p_matrices[name] = torch.empty(
                    d,
                    rank,
                    dtype=mat.dtype,
                    device=mat.device,
                )

            q = self.q_matrices[name]
            p = self.p_matrices[name]
            # Power iteration.
            for _ in range(self.num_power_iters):
                p = _orthogonalize(mat @ q)
                q = mat.T @ p
            # Stash for next round (warm start).
            self.p_matrices[name] = p
            self.q_matrices[name] = q

            if self.use_error_feedback:
                reconstructed = p @ q.T
                self.residual_buffers[name] = mat - reconstructed

            out[name] = {
                "P": p,
                "Q": q,
                "shape": original_shape,
                "rank": rank,
                "method": "powersgd_standalone",
            }
        return out

    def decompress(
        self, compressed: dict[str, dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Reconstruct gradients from their (P, Q) factors."""

        out: dict[str, torch.Tensor] = {}
        for name, payload in compressed.items():
            p = payload["P"]
            q = payload["Q"]
            shape = payload["shape"]
            approx = p @ q.T
            if len(shape) != 2:
                approx = approx.reshape(shape)
            out[name] = approx
        return out

    def compression_ratio(self, gradient: torch.Tensor) -> float:
        """Return uncompressed_size / compressed_size for a single tensor."""

        if gradient.numel() == 0:
            return 1.0
        mat = self._as_matrix(gradient)
        d, m = mat.shape
        r = min(self.rank, d, m)
        uncompressed = d * m
        compressed = d * r + r * m
        if compressed == 0:
            return float("inf")
        return uncompressed / compressed

    # ----- internals ----------------------------------------------------- #

    @staticmethod
    def _as_matrix(grad: torch.Tensor) -> torch.Tensor:
        """View an arbitrary-rank gradient as a 2-D matrix."""

        if grad.dim() == 0:
            return grad.view(1, 1)
        if grad.dim() == 1:
            return grad.view(1, -1)
        if grad.dim() == 2:
            return grad
        return grad.reshape(grad.shape[0], -1)


__all__ = [
    "POWERSGD_AVAILABLE",
    "PowerSGDCompressor",
    "install_powersgd",
]
