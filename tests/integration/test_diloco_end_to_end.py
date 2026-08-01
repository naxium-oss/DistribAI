"""End-to-end integration test for the DiLoCo outer-step path.

Spins up an in-process :class:`DiLoCoCoordinator` plus three mock worker
daemons (no gRPC stream, no S3 -- just direct method calls). Each worker
holds a copy of a tiny 2-layer model, the coordinator holds the canonical
weights. Across 5 outer rounds we assert:

  * global loss decreases monotonically (allowing one outlier round to
    cover stochastic minibatch noise);
  * the coordinator's ``round_id`` advances by exactly one per outer step;
  * the canonical weights remain shape-consistent throughout.

This is the integration that catches wiring regressions between
:class:`DiLoCoOuterStep`, :class:`DiLoCoCoordinator`, and the per-worker
runner contract (snapshot theta_start, run inner loop, submit pseudo-grad,
receive new canonical weights).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

from services_python.diloco import DiLoCoCoordinator

# --------------------------------------------------------- tiny shared model


def _build_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )


def _model_to_ndarrays(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        name: param.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, param in model.state_dict().items()
    }


def _load_ndarrays(model: nn.Module, weights: dict[str, np.ndarray]) -> None:
    new_sd = {}
    for name, ref in model.state_dict().items():
        arr = weights[name]
        new_sd[name] = torch.as_tensor(arr, dtype=ref.dtype, device=ref.device).reshape(
            ref.shape
        )
    model.load_state_dict(new_sd, strict=True)


def _eval_loss(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:  # noqa: N803
    model.eval()
    with torch.no_grad():
        return float(nn.functional.mse_loss(model(X), y).item())


# --------------------------------------------------------- mock worker daemon


@dataclass
class _MockWorker:
    worker_id: str
    model: nn.Module
    optimizer: torch.optim.Optimizer
    X: torch.Tensor
    y: torch.Tensor

    def run_round(
        self,
        canonical: dict[str, np.ndarray],
        H: int,  # noqa: N803
        batch_size: int = 8,
    ) -> dict[str, np.ndarray]:
        """Snapshot canonical, run H steps, return pseudo-grad ndarrays."""
        _load_ndarrays(self.model, canonical)
        theta_start = _model_to_ndarrays(self.model)
        self.model.train()
        n = self.X.shape[0]
        for _ in range(H):
            idx = torch.randint(0, n, (batch_size,))
            self.optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(self.model(self.X[idx]), self.y[idx])
            loss.backward()
            self.optimizer.step()
        theta_after = _model_to_ndarrays(self.model)
        return {k: (theta_start[k] - theta_after[k]).astype(np.float32) for k in theta_start}


# ------------------------------------------------------------- the actual test


@pytest.mark.asyncio
async def test_diloco_end_to_end_3_workers_5_rounds() -> None:
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    # Shared synthetic regression task: y = relu(W1 x + b1) . W2 + b2,
    # with each worker getting a distinct data shard but a shared eval set.
    d_in = 4
    n_per_worker = 256

    # Ground-truth model used to generate targets.
    ground_truth = _build_model(seed=999)
    with torch.no_grad():
        x_eval = torch.tensor(rng.normal(size=(500, d_in)).astype(np.float32))
        y_eval = ground_truth(x_eval).detach()

    worker_data = []
    for _wid in range(3):
        features = torch.tensor(rng.normal(size=(n_per_worker, d_in)).astype(np.float32))
        with torch.no_grad():
            y = ground_truth(features)
        worker_data.append((features, y))

    # Each worker has its own model + optimiser; the coordinator owns
    # the canonical weights.
    workers: list[_MockWorker] = []
    for wid, (features, labels) in enumerate(worker_data):
        model = _build_model(seed=0)  # all start identical, but that gets overridden
        optim = torch.optim.AdamW(model.parameters(), lr=5e-3)
        workers.append(_MockWorker(f"w-{wid}", model, optim, features, labels))

    # Coordinator initialised with the same initial weights as worker 0.
    initial_weights = _model_to_ndarrays(workers[0].model)
    coordinator = DiLoCoCoordinator()
    await coordinator.register_job(
        "e2e-job",
        initial_weights,
        outer_lr=0.7,
        outer_momentum=0.9,
        H=25,
        min_workers=3,
    )

    losses: list[float] = []
    # Initial loss (before any outer steps): all workers identical to canonical.
    eval_model = _build_model(seed=0)
    _load_ndarrays(eval_model, coordinator.get_current_weights("e2e-job"))
    losses.append(_eval_loss(eval_model, x_eval, y_eval))

    n_rounds = 5
    for round_idx in range(n_rounds):
        canonical = coordinator.get_current_weights("e2e-job")
        assert canonical is not None
        # Each worker runs H=25 inner steps and submits its pseudo-grad.
        for w in workers:
            pseudo = w.run_round(canonical, H=25)
            ok = await coordinator.submit_pseudo_gradient("e2e-job", w.worker_id, pseudo)
            assert ok, f"submit failed for {w.worker_id}"

        # Trigger the outer step. round_id should advance by exactly 1.
        before_round = coordinator.round_id("e2e-job")
        result = await coordinator.maybe_aggregate("e2e-job")
        assert result is not None, f"round {round_idx}: aggregate returned None"
        new_weights, completed_round = result
        assert completed_round == before_round
        assert coordinator.round_id("e2e-job") == before_round + 1

        # Shapes preserved across the outer step.
        for name, init in initial_weights.items():
            assert new_weights[name].shape == init.shape, (
                f"shape changed for {name}: {new_weights[name].shape} vs {init.shape}"
            )

        # Eval new canonical weights.
        _load_ndarrays(eval_model, new_weights)
        losses.append(_eval_loss(eval_model, x_eval, y_eval))

    # Sanity: round_id advanced by n_rounds.
    assert coordinator.round_id("e2e-job") == n_rounds

    # Loss decreased monotonically allowing for one outlier round.
    # We test: final loss < initial loss AND at most one round
    # increased the loss.
    deltas = np.diff(losses)
    n_increases = int((deltas > 0).sum())
    assert losses[-1] < losses[0], (
        f"final loss {losses[-1]:.4f} >= initial loss {losses[0]:.4f} "
        f"(curve: {losses})"
    )
    assert n_increases <= 1, (
        f"loss increased in {n_increases} rounds (allowed: 1). curve: {losses}"
    )


@pytest.mark.asyncio
async def test_diloco_admin_trigger_endpoint_advances_round() -> None:
    """trigger_aggregate (admin endpoint hook) must advance the round."""
    coordinator = DiLoCoCoordinator()
    await coordinator.register_job(
        "admin-job",
        {"w": np.ones(3, dtype=np.float32)},
        min_workers=5,  # higher than what we submit -> maybe_aggregate would skip
    )
    # Submit just one worker.
    await coordinator.submit_pseudo_gradient(
        "admin-job", "lone-worker", {"w": np.full(3, 0.1, dtype=np.float32)}
    )

    # maybe_aggregate won't fire (1 < 5).
    assert await coordinator.maybe_aggregate("admin-job") is None
    assert coordinator.round_id("admin-job") == 0

    # trigger_aggregate forces the step.
    result = await coordinator.trigger_aggregate("admin-job")
    assert result is not None
    new_weights, round_id = result
    assert round_id == 0
    assert coordinator.round_id("admin-job") == 1

    # trigger_aggregate with no pending submissions returns None.
    assert await coordinator.trigger_aggregate("admin-job") is None
    assert coordinator.round_id("admin-job") == 1  # unchanged


@pytest.mark.asyncio
async def test_diloco_unregister_removes_state() -> None:
    coordinator = DiLoCoCoordinator()
    await coordinator.register_job("temp", {"w": np.zeros(2)}, min_workers=1)
    assert coordinator.has_job("temp")
    await coordinator.unregister_job("temp")
    assert not coordinator.has_job("temp")
    assert coordinator.get_current_weights("temp") is None
    assert coordinator.round_id("temp") is None
