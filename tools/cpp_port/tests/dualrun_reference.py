#!/usr/bin/env python
"""Python reference for the dual-run diff machine.

Runs the baseline sandbox child semantics (train 1K MLP under the project
rlimits, in a forked child with setrlimit applied) and emits the SAME
envelope JSON as tools/cpp_port/apps/dualrun.cpp so the comparer can diff
them key by key.

Usage: .venv/bin/python tools/cpp_port/tests/dualrun_reference.py --seed 42 --steps 200
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "worker" / "src" / "distribai_proto"))


def _child(queue, seed: int, steps: int) -> None:
    """Runs in a forked child with the mandatory rlimits applied."""
    import resource

    mem_bytes = int(os.getenv("BENCH_MEM_MB", 4096)) * 1024 * 1024
    cpu = int(os.getenv("BENCH_CPU_SEC", 300))
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(10, 30), nn.ReLU(), nn.Linear(30, 30), nn.ReLU(), nn.Linear(30, 1)
        )
        n_params = sum(p.numel() for p in model.parameters())
        opt = optim.AdamW(model.parameters(), lr=0.01)
        crit = nn.MSELoss()

        t0 = time.perf_counter()
        losses = []
        for _ in range(steps):
            x = torch.rand(64, 10)
            y = torch.rand(64, 1)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        wall = time.perf_counter() - t0

        grads = [
            p.grad.detach().flatten().tolist()  # type: ignore[union-attr]
            for p in model.parameters()
            if p.grad is not None
        ]
        flat = [v for g in grads for v in g]
        r6 = lambda v: round(v, 6)  # noqa: E731
        queue.put(
            {
                "status": "ok",
                "n_params": n_params,
                "wall_s": round(wall, 6),
                "steps_per_s": round(steps / wall, 1),
                "ms_per_step": round(1000.0 * wall / steps, 3),
                "final_loss": r6(losses[-1]),
                "first5_losses": [r6(v) for v in losses[:5]],
                "grad_len": len(flat),
                "grad_sum": sum(flat),
                "grad_first3": [r6(v) for v in flat[:3]],
            }
        )
    except Exception as e:  # noqa: BLE001
        queue.put({"status": "error", "error": f"{type(e).__name__}: {e}"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()

    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(queue, args.seed, args.steps))
    proc.start()
    proc.join(180)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        print(json.dumps({"status": "timeout"}))
        return 1
    result = (
        queue.get() if not queue.empty() else {"status": "crash", "error": f"exit={proc.exitcode}"}
    )
    print(json.dumps(result))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
