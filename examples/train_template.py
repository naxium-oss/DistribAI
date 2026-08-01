#!/usr/bin/env python3
"""Run a real supervised PyTorch regression job for DistribAI.

Provide ``config.json`` and ``train.json`` beside this file. ``train.json``
must contain ``{"features": [[...], ...], "targets": [[...], ...]}``.
The example intentionally fails when data or PyTorch is unavailable instead of
claiming that training completed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent


def _load_json(filename: str) -> dict[str, Any]:
    path = ROOT / filename
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return payload


def load_config() -> dict[str, Any]:
    """Load training configuration from ``config.json``."""
    return _load_json("config.json")


def load_hyperparams() -> dict[str, Any]:
    """Load optimizer parameters from ``hyperparams.json``."""
    return _load_json("hyperparams.json")


def train(config: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Fit a linear model to the operator-provided regression records."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Training requires torch; install the worker dependencies") from exc

    dataset_path = ROOT / str(config.get("dataset", "train.json"))
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")
    with dataset_path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, dict):
        raise ValueError("Training dataset must be a JSON object")

    features = torch.tensor(dataset.get("features", []), dtype=torch.float32)
    targets = torch.tensor(dataset.get("targets", []), dtype=torch.float32)
    if features.ndim != 2 or targets.ndim not in (1, 2) or features.size(0) == 0:
        raise ValueError("Dataset must contain non-empty 2-D features and matching targets")
    if targets.ndim == 1:
        targets = targets.unsqueeze(1)
    if targets.size(0) != features.size(0):
        raise ValueError("features and targets must contain the same number of rows")

    model = nn.Linear(features.size(1), targets.size(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params.get("lr", 1e-2)))
    criterion = nn.MSELoss()
    steps = int(config.get("total_steps", params.get("steps", 100)))
    if steps < 1:
        raise ValueError("total_steps must be positive")

    losses: list[float] = []
    model.train()
    for _ in range(steps):
        prediction = model(features)
        loss = criterion(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    checkpoint_path = ROOT / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    return {
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "steps_completed": steps,
        "samples": int(features.size(0)),
        "input_features": int(features.size(1)),
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    config = load_config()
    params = load_hyperparams()
    metrics = train(config, params)
    task_id = os.getenv("DISTRIBAI_TASK_ID", "unknown")
    job_id = os.getenv("DISTRIBAI_JOB_ID", "unknown")
    output = {"status": "completed", "task_id": task_id, "job_id": job_id, "metrics": metrics}
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Training failed: {exc}")
        raise SystemExit(1) from exc
