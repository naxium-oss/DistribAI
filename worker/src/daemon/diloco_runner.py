"""Worker-side DiLoCo outer-step runner (arxiv:2311.08105).

On the worker, the DiLoCo runner is responsible for:

  1. At round start (``DiLoCoRoundStart``), download the canonical weights
     from the orchestrator and snapshot them as ``theta_start``.
  2. Run ``H`` inner SGD steps with the configured inner optimiser
     (AdamW by default, but any registered optimiser via the factory).
  3. Compute the pseudo-gradient ``theta_start - theta_current`` per
     parameter tensor.
  4. Upload the pseudo-grad blob and send ``DiLoCoPseudoGradient`` to
     the orchestrator.
  5. Wait for ``DiLoCoRoundComplete``, replace the local weights with
     the new canonical ones, and start the next round.

The inner optimiser is resolved at runtime through
:func:`get_inner_optimizer_factory` -- this is a deliberate registration
point so the AuON branch (sibling worktree ``wt-auon-powersgd``) can add
itself without this module taking a hard dependency on AuON.

Reference: https://arxiv.org/abs/2311.08105
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- factory

# Type alias for an optimiser factory: takes the parameter iterable and the
# per-job hyperparams dict and returns a torch.optim.Optimizer.
OptimizerFactory = Callable[[Iterable[torch.nn.Parameter], dict[str, Any]], torch.optim.Optimizer]


_OPTIMIZER_REGISTRY: dict[str, OptimizerFactory] = {}


def register_inner_optimizer(name: str, factory: OptimizerFactory) -> None:
    """Register an inner-optimiser factory under ``name``.

    Called at module import time by the AuON / PowerSGD / etc. plugin
    modules. Names are case-insensitive and stored lower-cased.

    Args:
        name: Short identifier (e.g. ``"adamw"``, ``"auon"``).
        factory: Callable mapping ``(params, hyperparams)`` to an
            instantiated :class:`torch.optim.Optimizer`.

    Raises:
        ValueError: If ``name`` is empty or whitespace.
    """
    key = (name or "").strip().lower()
    if not key:
        raise ValueError("optimizer name must be non-empty")
    if key in _OPTIMIZER_REGISTRY:
        logger.debug("Re-registering inner optimizer %r (replacing prior factory)", key)
    _OPTIMIZER_REGISTRY[key] = factory


def get_inner_optimizer_factory(name: str | None) -> OptimizerFactory:
    """Resolve ``name`` to a registered optimiser factory.

    Args:
        name: Registered factory name. Falsy values fall back to ``"adamw"``.

    Returns:
        The factory callable.

    Raises:
        KeyError: If ``name`` is not registered. The error message lists
            the currently-registered names to aid debugging.
    """
    key = (name or "adamw").strip().lower()
    if key not in _OPTIMIZER_REGISTRY:
        raise KeyError(
            f"Inner optimizer {key!r} is not registered. "
            f"Known: {sorted(_OPTIMIZER_REGISTRY)}. "
            f"Plugin modules must call register_inner_optimizer() at import."
        )
    return _OPTIMIZER_REGISTRY[key]


def _default_adamw_factory(
    params: Iterable[torch.nn.Parameter],
    hyperparams: dict[str, Any],
) -> torch.optim.Optimizer:
    """Standard AdamW with paper defaults for DiLoCo's inner loop."""
    lr = float(hyperparams.get("inner_lr", 4e-4))
    betas = tuple(hyperparams.get("inner_betas", (0.9, 0.95)))
    weight_decay = float(hyperparams.get("inner_weight_decay", 0.1))
    eps = float(hyperparams.get("inner_eps", 1e-8))
    return torch.optim.AdamW(
        list(params),
        lr=lr,
        betas=betas,  # type: ignore[arg-type]
        weight_decay=weight_decay,
        eps=eps,
    )


# Register the default. Plugin modules (e.g. AuON in wt-auon-powersgd)
# register additional factories at their own import time.
register_inner_optimizer("adamw", _default_adamw_factory)


# ----------------------------------------------------------------- runner


class _BlobIO(Protocol):
    """Subset of :class:`~worker.src.daemon.s3_util` we depend on.

    Defined as a Protocol so tests can pass an in-memory stub without
    touching the real S3 client.
    """

    def download_weights(self, url: str) -> dict[str, torch.Tensor]: ...

    def upload_pseudo_gradient(
        self,
        job_id: str,
        round_id: int,
        worker_id: str,
        pseudo_grad: dict[str, torch.Tensor],
    ) -> str: ...


@dataclass
class DiLoCoRunnerConfig:
    """Per-job DiLoCo configuration on the worker.

    Mirrors ``DiLoCoConfig`` in the proto (kept as a Python dataclass so
    the runner can be unit-tested without spinning a real gRPC stream).
    """

    job_id: str
    worker_id: str
    H: int = 500
    inner_optimizer: str = "adamw"
    hyperparams: dict[str, Any] = field(default_factory=dict)


class DiLoCoRunner:
    """Worker-side DiLoCo loop driver.

    Owns the local model and the inner optimiser. The orchestrator pushes
    ``round_start`` / ``round_complete`` events and the runner blocks on a
    ``data_iter`` for inner-loop batches.

    Attributes:
        model: The local model being trained (state-dict-compatible).
        cfg: Per-job DiLoCo configuration.
        blob_io: Object responsible for download/upload of weight blobs.
        device: Torch device for the inner loop.
        round_id: Current outer round (advances on each ``round_complete``).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DiLoCoRunnerConfig,
        blob_io: _BlobIO,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.blob_io = blob_io
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.model.to(self.device)

        factory = get_inner_optimizer_factory(cfg.inner_optimizer)
        self._optimizer: torch.optim.Optimizer = factory(
            self.model.parameters(), cfg.hyperparams
        )
        self.round_id: int = 0
        # theta_start is captured at the beginning of each round.
        self._theta_start: dict[str, torch.Tensor] | None = None

    # -------------------------------------------------------------- public

    def on_round_start(self, round_id: int, weights_blob_url: str) -> None:
        """Handle ``DiLoCoRoundStart`` from the orchestrator.

        Downloads the canonical weights, loads them into the local model,
        snapshots them as ``theta_start``, and rebuilds the inner optimiser
        state so per-round optimiser state cannot leak across rounds.
        """
        logger.info(
            "DiLoCo round %d starting (job=%s worker=%s, fetching %s)",
            round_id,
            self.cfg.job_id,
            self.cfg.worker_id,
            weights_blob_url,
        )
        canonical = self.blob_io.download_weights(weights_blob_url)
        self._load_weights(canonical)
        self.round_id = round_id
        # Snapshot a CPU-detached copy so the pseudo-grad math is GPU-agnostic.
        self._theta_start = {
            name: param.detach().clone().cpu()
            for name, param in self.model.state_dict().items()
        }
        # Fresh inner optimiser per round -- DiLoCo paper resets inner state.
        factory = get_inner_optimizer_factory(self.cfg.inner_optimizer)
        self._optimizer = factory(self.model.parameters(), self.cfg.hyperparams)

    def run_inner_loop(
        self,
        data_iter: Iterator[tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        """Run exactly ``H`` inner steps. Returns mean inner loss."""
        if self._theta_start is None:
            raise RuntimeError(
                "DiLoCoRunner.run_inner_loop called before on_round_start"
            )
        self.model.train()
        total_loss = 0.0
        steps = 0
        for step in range(self.cfg.H):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                logger.warning(
                    "DiLoCo round %d: data_iter exhausted after %d/%d steps",
                    self.round_id,
                    step,
                    self.cfg.H,
                )
                break
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self._optimizer.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self._optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            steps += 1
        mean_loss = total_loss / max(steps, 1)
        logger.info(
            "DiLoCo round %d: inner loop done (%d steps, mean loss %.4f)",
            self.round_id,
            steps,
            mean_loss,
        )
        return mean_loss

    def compute_pseudo_gradient(self) -> dict[str, torch.Tensor]:
        """Return ``theta_start - theta_current`` per parameter.

        Computed on CPU in float32 to keep aggregation deterministic
        regardless of worker GPU dtype / mixed precision.
        """
        if self._theta_start is None:
            raise RuntimeError(
                "DiLoCoRunner.compute_pseudo_gradient called before on_round_start"
            )
        current = self.model.state_dict()
        pseudo: dict[str, torch.Tensor] = {}
        for name, start in self._theta_start.items():
            cur = current[name].detach().cpu().to(torch.float32)
            pseudo[name] = (start.to(torch.float32) - cur).contiguous()
        return pseudo

    def submit_pseudo_gradient(self) -> str:
        """Upload the pseudo-grad blob and return its URL.

        The runner does *not* itself construct the
        :class:`DiLoCoPseudoGradient` protobuf -- that is the daemon's
        responsibility (the daemon already owns the gRPC stream). The
        runner returns the blob URL the daemon should put on the wire.
        """
        pseudo = self.compute_pseudo_gradient()
        url = self.blob_io.upload_pseudo_gradient(
            job_id=self.cfg.job_id,
            round_id=self.round_id,
            worker_id=self.cfg.worker_id,
            pseudo_grad=pseudo,
        )
        logger.info(
            "DiLoCo round %d: uploaded pseudo-grad (%d tensors) -> %s",
            self.round_id,
            len(pseudo),
            url,
        )
        return url

    def on_round_complete(self, round_id: int, new_weights_blob_url: str) -> None:
        """Handle ``DiLoCoRoundComplete``: replace local weights for next round."""
        logger.info(
            "DiLoCo round %d complete (job=%s, fetching new canonical weights %s)",
            round_id,
            self.cfg.job_id,
            new_weights_blob_url,
        )
        new_weights = self.blob_io.download_weights(new_weights_blob_url)
        self._load_weights(new_weights)
        # Clear theta_start so a misuse (running inner_loop before next
        # on_round_start) fails loudly instead of computing a stale pseudo-grad.
        self._theta_start = None

    # ------------------------------------------------------------- helpers

    def _load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load ``weights`` (CPU tensors) into the model on ``self.device``."""
        current = self.model.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        for name, ref in current.items():
            if name not in weights:
                raise KeyError(
                    f"DiLoCo: canonical weights missing parameter {name!r}"
                )
                # (KeyError is raised so retries / re-syncs surface the bug)
            t = weights[name]
            if t.shape != ref.shape:
                raise ValueError(
                    f"DiLoCo: canonical weight {name!r} shape {tuple(t.shape)} "
                    f"!= local shape {tuple(ref.shape)}"
                )
            loaded[name] = t.to(device=self.device, dtype=ref.dtype)
        self.model.load_state_dict(loaded, strict=True)


__all__ = [
    "DiLoCoRunner",
    "DiLoCoRunnerConfig",
    "OptimizerFactory",
    "get_inner_optimizer_factory",
    "register_inner_optimizer",
]
