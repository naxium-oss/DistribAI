"""Real PyTorch smoke job for DistribAI script-job rehearsals."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        print(f"PyTorch is required for this smoke job: {exc}", file=sys.stderr)
        return 1

    steps = int(os.environ.get("DISTRIBAI_STEPS", "3"))
    if steps < 1:
        print("DISTRIBAI_STEPS must be positive", file=sys.stderr)
        return 1
    torch.manual_seed(0)
    features = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    targets = 2 * features + 1
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    losses: list[float] = []
    for index in range(steps):
        loss = criterion(model(features), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        print(f"step {index + 1}/{steps} loss={value:.6f}")

    metrics = {
        "ok": True,
        "steps": steps,
        "framework": "pytorch",
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    out = os.environ.get("DISTRIBAI_RESULTS_PATH", "results.json")
    Path(out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
