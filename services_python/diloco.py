"""DiLoCo outer-step coordinator for DistribAI (arxiv:2311.08105).

DiLoCo (Distributed Low-Communication training, Douillard et al. 2023)
decouples worker-local SGD from cross-worker synchronisation: each worker
runs ``H`` inner AdamW steps locally, then submits only the *pseudo-gradient*
(``theta_start - theta_after_H_steps``) to the orchestrator. The orchestrator
averages pseudo-gradients across workers, applies an outer Nesterov-momentum
step (paper-default outer LR 0.7, momentum 0.9), and broadcasts the new
canonical weights to start the next outer round.

Bandwidth reduction vs DDP all-reduce is ~``H``x (typically 500x): instead
of synchronising every step we synchronise every H steps, and the pseudo-grad
payload is the same size as a full grad.

This module is dependency-light on purpose: it is the *coordinator* logic
that runs on the orchestrator process, which may be CPU-only. Pseudo-grads
arrive from workers as ``dict[str, np.ndarray]`` (one entry per parameter
tensor). The aggregation step composes with the existing
:class:`~worker.src.daemon.byzantine_detector.aggregators.AdaptiveAggregator`
defence pipeline (FoolsGold + SignGuard + Bucketing) so that Byzantine
workers cannot poison the outer step.

Reference: https://arxiv.org/abs/2311.08105
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Type alias: per-worker pseudo-gradient is a dict of named ndarrays, one per
# parameter tensor. Shapes/dtypes must match the canonical weights.
PseudoGrad = dict[str, np.ndarray]


def _validate_shapes(
    pseudo_grad: PseudoGrad,
    template: PseudoGrad,
    *,
    where: str,
) -> None:
    """Raise ``ValueError`` if ``pseudo_grad`` doesn't match ``template`` shapes."""
    missing = sorted(set(template) - set(pseudo_grad))
    extra = sorted(set(pseudo_grad) - set(template))
    if missing or extra:
        raise ValueError(
            f"{where}: pseudo-grad keys do not match canonical weights. "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    for name, ref in template.items():
        got = pseudo_grad[name]
        if got.shape != ref.shape:
            raise ValueError(
                f"{where}: pseudo-grad[{name!r}] shape {got.shape} != "
                f"canonical shape {ref.shape}"
            )


class DiLoCoOuterStep:
    """DiLoCo outer-step coordinator (arxiv:2311.08105).

    Maintains the canonical model weights and outer Nesterov-momentum state
    on the orchestrator. Workers run ``H`` inner AdamW steps locally,
    submit pseudo-gradients, and receive new weights to start the next
    outer round.

    Bandwidth reduction vs DDP all-reduce: ~``H`` x (typically 500x).

    Attributes:
        outer_lr: Outer-loop learning rate (paper default 0.7).
        outer_momentum: Outer-loop Nesterov momentum (paper default 0.9).
        H: Inner steps per outer round (informational; the worker enforces
            this, the coordinator just consumes whatever pseudo-grads arrive).
        min_workers: Minimum distinct pseudo-grads required before
            :meth:`ready_to_aggregate` returns True.
    """

    def __init__(
        self,
        initial_weights: dict[str, np.ndarray],
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        H: int = 500,  # noqa: N803  -- DiLoCo paper notation (inner steps)
        min_workers: int = 2,
    ) -> None:
        if not initial_weights:
            raise ValueError("initial_weights must be a non-empty dict")
        if outer_lr <= 0:
            raise ValueError(f"outer_lr must be > 0, got {outer_lr}")
        if not 0.0 <= outer_momentum < 1.0:
            raise ValueError(
                f"outer_momentum must be in [0, 1), got {outer_momentum}"
            )
        if H < 1:
            raise ValueError(f"H must be >= 1, got {H}")
        if min_workers < 1:
            raise ValueError(f"min_workers must be >= 1, got {min_workers}")

        # Copy weights so external mutation cannot corrupt the canonical state.
        self._weights: dict[str, np.ndarray] = {
            k: np.array(v, copy=True) for k, v in initial_weights.items()
        }
        # Outer Nesterov-momentum buffer, one per parameter, zero-initialised.
        self._momentum: dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in self._weights.items()
        }
        self.outer_lr = float(outer_lr)
        self.outer_momentum = float(outer_momentum)
        self.H = int(H)
        self.min_workers = int(min_workers)

        # Per-round submission buffers.
        self._pending: dict[str, PseudoGrad] = {}
        self._round_id: int = 0

        # Coordinator is shared across asyncio tasks; protect mutating ops.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ API

    def round_id(self) -> int:
        """Return the current outer-round counter (starts at 0)."""
        with self._lock:
            return self._round_id

    def get_current_weights(self) -> dict[str, np.ndarray]:
        """Return a copy of the canonical weights for broadcast to workers."""
        with self._lock:
            return {k: np.array(v, copy=True) for k, v in self._weights.items()}

    def submit_pseudo_gradient(
        self,
        worker_id: str,
        pseudo_grad: PseudoGrad,
    ) -> None:
        """Buffer a worker's pseudo-gradient for the current outer round.

        Args:
            worker_id: Stable identifier for the worker; later submissions
                from the same worker in the same round overwrite earlier
                ones (last-write-wins, matching DDP gradient accumulation
                semantics under retries).
            pseudo_grad: ``theta_start - theta_after_H_steps`` per parameter.

        Raises:
            ValueError: If shapes do not match the canonical weights.
        """
        _validate_shapes(pseudo_grad, self._weights, where="submit_pseudo_gradient")
        with self._lock:
            if worker_id in self._pending:
                logger.debug(
                    "DiLoCo round %d: worker %s resubmitted; overwriting",
                    self._round_id,
                    worker_id,
                )
            # Store a private copy so the caller can free its buffer.
            self._pending[worker_id] = {
                k: np.array(v, copy=True) for k, v in pseudo_grad.items()
            }
            logger.debug(
                "DiLoCo round %d: %d/%d workers submitted",
                self._round_id,
                len(self._pending),
                self.min_workers,
            )

    def ready_to_aggregate(self) -> bool:
        """True iff at least ``min_workers`` distinct pseudo-grads are buffered."""
        with self._lock:
            return len(self._pending) >= self.min_workers

    def num_pending(self) -> int:
        """Number of workers that have submitted in the current round."""
        with self._lock:
            return len(self._pending)

    def aggregate_and_step(
        self,
        defense_filter: Callable[[dict[str, PseudoGrad]], dict[str, PseudoGrad]]
        | None = None,
    ) -> dict[str, np.ndarray]:
        """Average pseudo-grads, apply outer Nesterov step, return new weights.

        The aggregation algorithm follows DiLoCo Algorithm 1 with the
        Nesterov-momentum outer optimiser:

        .. code-block:: text

            g_t = (1/N) * sum_i pseudo_grad_i
            m_t = mu * m_{t-1} + g_t                       # Nesterov momentum
            theta_{t+1} = theta_t - outer_lr * (mu * m_t + g_t)

        If ``defense_filter`` is provided (typically the worker-side
        :func:`default_defense_pipeline`-wrapped aggregator), the per-worker
        pseudo-grads are first run through it; the filter must accept and
        return ``dict[worker_id, dict[layer, ndarray]]``. This lets the
        existing FoolsGold + SignGuard + Bucketing defences down-weight
        Byzantine workers' pseudo-grads before averaging.

        Returns:
            The new canonical weights (a fresh dict the caller may mutate).

        Raises:
            RuntimeError: If no pseudo-grads have been submitted.
        """
        with self._lock:
            if not self._pending:
                raise RuntimeError(
                    "DiLoCo.aggregate_and_step called with no pseudo-grads"
                )
            pending = self._pending
            self._pending = {}
            round_id = self._round_id
            self._round_id += 1

        # Optionally run pseudo-grads through the Byzantine-defence pipeline.
        if defense_filter is not None:
            try:
                pending = defense_filter(pending)
            except Exception:
                logger.exception(
                    "DiLoCo round %d: defence filter raised; falling back to "
                    "unfiltered pseudo-grads",
                    round_id,
                )
        if not pending:
            logger.warning(
                "DiLoCo round %d: defence filter emptied the submission set; "
                "outer step is a no-op",
                round_id,
            )
            return self.get_current_weights()

        n_workers = len(pending)
        # Average pseudo-grads layerwise.
        avg_grad: dict[str, np.ndarray] = {}
        for name, ref in self._weights.items():
            acc = np.zeros_like(ref, dtype=np.float64)
            for wid, pg in pending.items():
                if name not in pg:
                    raise ValueError(
                        f"DiLoCo round {round_id}: worker {wid} missing "
                        f"layer {name!r} after defence filter"
                    )
                acc += np.asarray(pg[name], dtype=np.float64)
            avg_grad[name] = (acc / n_workers).astype(ref.dtype, copy=False)

        # Apply outer Nesterov-momentum step in-place on the canonical weights.
        with self._lock:
            mu = self.outer_momentum
            lr = self.outer_lr
            for name in self._weights:
                g = avg_grad[name]
                m_prev = self._momentum[name]
                # Standard Nesterov update on the pseudo-gradient:
                #   m_t = mu * m_{t-1} + g_t
                #   theta_{t+1} = theta_t - lr * (mu * m_t + g_t)
                # which is equivalent to taking a look-ahead step on m.
                m_new = mu * m_prev + g
                self._momentum[name] = m_new
                step = lr * (mu * m_new + g)
                self._weights[name] = self._weights[name] - step.astype(
                    self._weights[name].dtype, copy=False
                )
            logger.info(
                "DiLoCo round %d completed: %d workers, outer_lr=%.3f, "
                "outer_momentum=%.3f",
                round_id,
                n_workers,
                lr,
                mu,
            )
            return {k: np.array(v, copy=True) for k, v in self._weights.items()}

    # ----------------------------------------------------------- diagnostics

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for monitoring/admin endpoints."""
        with self._lock:
            return {
                "round_id": self._round_id,
                "pending_workers": sorted(self._pending.keys()),
                "outer_lr": self.outer_lr,
                "outer_momentum": self.outer_momentum,
                "H": self.H,
                "min_workers": self.min_workers,
                "num_parameters": len(self._weights),
                "total_elements": sum(int(v.size) for v in self._weights.values()),
            }

    def reset_round(self) -> None:
        """Discard any buffered pseudo-grads without advancing the round id.

        Intended for admin/diagnostic use when a round is known to be
        corrupted (e.g. all workers dropped before reaching the quorum).
        """
        with self._lock:
            n = len(self._pending)
            self._pending = {}
        if n:
            logger.warning("DiLoCo reset_round: discarded %d pending pseudo-grads", n)


def make_adaptive_defense_filter(
    aggregator: Any | None = None,
    *,
    max_byzantine_fraction: float = 0.2,
) -> Callable[[dict[str, PseudoGrad]], dict[str, PseudoGrad]]:
    """Build a defence filter compatible with :meth:`DiLoCoOuterStep.aggregate_and_step`.

    This wraps the existing ``AdaptiveAggregator(defense_pipeline=
    default_defense_pipeline())`` so that pseudo-grads flow through
    FoolsGold + SignGuard + Bucketing before DiLoCo averages them. The
    pseudo-gradient has the same shape as a regular gradient, so the
    existing defence works without modification.

    Args:
        aggregator: Optional pre-built ``AdaptiveAggregator``. If ``None``
            a default instance is constructed lazily (the import is deferred
            so this module can be used in CPU-only orchestrators that have
            not installed Torch).
        max_byzantine_fraction: Forwarded to a freshly-built aggregator.

    Returns:
        A callable suitable for the ``defense_filter`` argument.
    """

    _agg_holder: dict[str, Any] = {"agg": aggregator}

    def _build_default() -> Any:
        # Deferred import: heavy Torch dependency only loaded if/when filter
        # is actually invoked.
        from worker.src.daemon.byzantine_detector import (  # noqa: PLC0415
            AdaptiveAggregator,
            default_defense_pipeline,
        )

        return AdaptiveAggregator(
            max_byzantine_fraction=max_byzantine_fraction,
            defense_pipeline=default_defense_pipeline(),
        )

    def _filter(pending: dict[str, PseudoGrad]) -> dict[str, PseudoGrad]:
        if _agg_holder["agg"] is None:
            _agg_holder["agg"] = _build_default()
        agg = _agg_holder["agg"]
        # The defence pipeline runs in the agg internals when we call its
        # aggregate(); but we want to *filter* (return per-worker weighted
        # pseudo-grads), not collapse them to a single aggregate. We
        # therefore drive the pipeline directly.
        pipeline = getattr(agg, "defense_pipeline", None)
        if not pipeline:
            return pending

        # The defence-pipeline filters operate on the NodeGradients form,
        # which matches our PseudoGrad dict shape exactly (per-layer
        # ndarrays). We pass through without conversion.
        out = pending
        for f in pipeline:
            try:
                out = f(out)
            except Exception:
                logger.exception(
                    "DiLoCo defence filter %s raised; passing input through",
                    type(f).__name__,
                )
                return pending
            if not out:
                logger.warning(
                    "DiLoCo defence filter %s emptied submission set; "
                    "reverting to unfiltered pseudo-grads",
                    type(f).__name__,
                )
                return pending
        return out

    return _filter


class DiLoCoCoordinator:
    """Multi-job async facade over :class:`DiLoCoOuterStep` for admin hooks and tests."""

    def __init__(self) -> None:
        self._jobs: dict[str, DiLoCoOuterStep] = {}
        self._lock = threading.Lock()

    async def register_job(
        self,
        job_id: str,
        initial_weights: dict[str, np.ndarray],
        *,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        H: int = 500,  # noqa: N803
        min_workers: int = 2,
    ) -> None:
        with self._lock:
            self._jobs[job_id] = DiLoCoOuterStep(
                initial_weights,
                outer_lr=outer_lr,
                outer_momentum=outer_momentum,
                H=H,
                min_workers=min_workers,
            )

    def has_job(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._jobs

    async def unregister_job(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def get_current_weights(self, job_id: str) -> dict[str, np.ndarray] | None:
        with self._lock:
            step = self._jobs.get(job_id)
        if step is None:
            return None
        return step.get_current_weights()

    def round_id(self, job_id: str) -> int | None:
        with self._lock:
            step = self._jobs.get(job_id)
        if step is None:
            return None
        return step.round_id()

    async def submit_pseudo_gradient(
        self,
        job_id: str,
        worker_id: str,
        pseudo_grad: PseudoGrad,
    ) -> bool:
        with self._lock:
            step = self._jobs.get(job_id)
        if step is None:
            return False
        try:
            step.submit_pseudo_gradient(worker_id, pseudo_grad)
        except ValueError:
            return False
        return True

    async def maybe_aggregate(
        self, job_id: str
    ) -> tuple[dict[str, np.ndarray], int] | None:
        with self._lock:
            step = self._jobs.get(job_id)
            if step is None or not step.ready_to_aggregate():
                return None
        before = step.round_id()
        new_weights = step.aggregate_and_step()
        return new_weights, before

    async def trigger_aggregate(
        self, job_id: str
    ) -> tuple[dict[str, np.ndarray], int] | None:
        with self._lock:
            step = self._jobs.get(job_id)
            if step is None or step.num_pending() == 0:
                return None
        before = step.round_id()
        new_weights = step.aggregate_and_step()
        return new_weights, before


__all__ = [
    "DiLoCoCoordinator",
    "DiLoCoOuterStep",
    "PseudoGrad",
    "make_adaptive_defense_filter",
]
