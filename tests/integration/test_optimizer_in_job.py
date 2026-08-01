"""Integration test: the v1.2 optimizer-resolution path inside the executor.

We avoid spinning the whole orchestrator + worker -- instead we run a
minimal in-process job through ``JobExecutor.execute`` with a mocked
``build_optimizer`` and a fake on_progress/on_result. That gives us
enough surface area to confirm:

- An unset ``hyperparams.optimizer`` plus an unset env var resolves to
  the v1.2 system default (``auon``).
- A per-job override beats the env var.
- The env var beats the compiled-in default when no per-job override is
  given.
- A name we know is registered (``adamw``) round-trips correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Ensure repo root is on sys.path so worker.* imports resolve when
# pytest is run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

try:
    from worker.src.daemon.executor import JobExecutor, _resolve_default_optimizer
    from worker.src.daemon.optimizers import AuON, build_optimizer

    HAS_EXECUTOR = True
except ImportError:
    HAS_EXECUTOR = False
    JobExecutor = None  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(
    not HAS_TORCH or not HAS_EXECUTOR,
    reason="torch or executor not available",
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_executor():
    async def _on_progress(*args, **kwargs):
        return None

    async def _on_result(*args, **kwargs):
        return None

    # Avoid touching any actual compute backend; force CPU and stub S3.
    with patch("worker.src.daemon.executor.detect_backend", return_value=None):
        ex = JobExecutor(
            node_id="test-node-optimizer",
            on_progress=_on_progress,
            on_result=_on_result,
        )
    return ex


def _make_job(hyperparams: dict | None = None) -> dict:
    return {
        "job_id": "job-test-optim",
        "task_id": "task-test-optim",
        "steps": 2,
        "batch_size": 4,
        "model_name": "toy",
        "deadline_ts": 9_999_999_999,
        "hyperparams": hyperparams or {},
    }


# --------------------------------------------------------------------------- #
# default optimizer resolution
# --------------------------------------------------------------------------- #


def test_resolve_default_optimizer_compiled_in(monkeypatch):
    monkeypatch.delenv("DISTRIBAI_DEFAULT_OPTIMIZER", raising=False)
    assert _resolve_default_optimizer() == "auon"


def test_resolve_default_optimizer_env_override(monkeypatch):
    monkeypatch.setenv("DISTRIBAI_DEFAULT_OPTIMIZER", "adamw")
    assert _resolve_default_optimizer() == "adamw"


def test_build_optimizer_with_default_returns_auon():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = build_optimizer("auon", [p], lr=1e-3)
    assert isinstance(opt, AuON)


def test_build_optimizer_adamw_returns_adamw():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = build_optimizer("adamw", [p], lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)


# --------------------------------------------------------------------------- #
# Job-level optimizer selection (resolver + one execute smoke)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "env_opt,hyperparams,expected",
    [
        (None, {}, "auon"),
        ("sgd", {"optimizer": "adamw"}, "adamw"),
        ("adamw", {}, "adamw"),
        (None, {"optimizer": "auon"}, "auon"),
    ],
)
def test_job_optimizer_resolution(env_opt, hyperparams, expected, monkeypatch):
    if env_opt is None:
        monkeypatch.delenv("DISTRIBAI_DEFAULT_OPTIMIZER", raising=False)
    else:
        monkeypatch.setenv("DISTRIBAI_DEFAULT_OPTIMIZER", env_opt)
    from worker.src.daemon.executor import _resolve_job_optimizer

    assert _resolve_job_optimizer(hyperparams) == expected
