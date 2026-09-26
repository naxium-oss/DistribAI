"""
Baseline 'before' benchmark for the C++ port mission.

Measures, on the current Python/PyTorch codebase:

1. Micro-workload floors (matmul, argmax, np+list serialization, allocator)
   -- these become the C++ comparison floors.
2. Training throughput at 1 / 1K / 10K / 100K params using the same
   regression task as worker/src/benchmark/bench_tensor.py.
3. **The mission workload**: training a ~1K-parameter model *inside* the
   project's own sandbox (worker/src/sandbox/sandbox.py, subprocess mode,
   Linux rlimits: RLIMIT_AS/RLIMIT_CPU/RLIMIT_FSIZE/RLIMIT_NPROC/RLIMIT_NOFILE).
   Emulates worker resource caps (executor.get_resource_limits) split across:
     - sandbox1x: one sandbox holding 100% of a pool (cpu_half x threads)
     - sandbox2x: two sandboxes holding 50% each
   Same total resources; sandboxes are isolated address spaces, so 2x also
   pays double torch import + double allocator arenas -- that is the
   per-sandbox overhead the C++ port targets.
4. End-to-end train->grad->JSON->parse handoff cycle, the real
   orchestrator/worker gradient path.

Usage:
    .venv/bin/python tools/bench/baseline_1k_train.py [--quick]

Writes runtime/baselines/baseline_python_<ts>.json (gitignored dir) and
prints a summary table. No code is modified -- evidence only.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


# ------------------------------------------------------------------
# Mandatory resource limits (hard requirements, not optional knobs).
# RLIMIT_AS is set EXACT (no +1 MB headroom) so every worker child is
# capped by its own address space; RLIMIT_CPU hard limit == soft limit.
# ------------------------------------------------------------------
def _uid_task_count() -> int:
    """Tasks (processes AND threads) already running under this UID.

    RLIMIT_NPROC counts threads, not just processes, and it is per-UID:
    on shared boxes you must count with `ps -L` or a fixed ceiling will
    starve torch's thread pool (libgomp: Thread creation failed).
    """
    try:
        out = subprocess.run(
            ["ps", "-u", str(os.getuid()), "-L", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return len([ln for ln in out.stdout.splitlines() if ln.strip()])
    except Exception:
        return 0


# NPROC is per-UID: on shared accounts a fixed value starves torch's
# thread pool (libgomp). Size it as current usage + headroom, with a floor.
_current_tasks = max(_uid_task_count(), 64)
SANDBOX_RLIMITS = {
    "mem_mb": int(os.getenv("BENCH_MEM_MB", 4096)),
    "cpu_sec": int(os.getenv("BENCH_CPU_SEC", 300)),
    "nproc": int(os.getenv("BENCH_NPROC", 0)) or (_current_tasks + 96),
    "nofile": 256,
    "fsize_mb": 512,
}
MIN_FREE_RAM_MB = int(os.getenv("BENCH_MIN_FREE_RAM_MB", 2048))


def _available_ram_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def preflight() -> None:
    """Abort before spawning anything if the box lacks headroom."""
    free = _available_ram_mb()
    print(
        f"preflight: available RAM {free} MiB (need >= {MIN_FREE_RAM_MB}), cores {os.cpu_count()}"
    )
    if free < MIN_FREE_RAM_MB:
        print(
            f"ABORT: only {free} MiB available (< {MIN_FREE_RAM_MB} MiB required). "
            "Free memory or lower BENCH_MEM_MB/BENCH_CPU_SEC.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def apply_strict_rlimits() -> None:
    """Apply hard rlimits to the CALLING process (used inside children)."""
    mem_bytes = SANDBOX_RLIMITS["mem_mb"] * 1024 * 1024
    cpu = SANDBOX_RLIMITS["cpu_sec"]
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))  # hard == soft
    resource.setrlimit(resource.RLIMIT_NPROC, (SANDBOX_RLIMITS["nproc"],) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (SANDBOX_RLIMITS["nofile"],) * 2)
    fsize = SANDBOX_RLIMITS["fsize_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "worker" / "src" / "distribai_proto"))

# Map worker repo layout to import paths used by services_python/worker code.
import importlib

import services_python
import worker

for pkg in (services_python, worker):
    sys.modules[pkg.__name__] = pkg

# The sandbox module imports "from worker.src.sandbox.serialization import ...";
# ensure that resolves inside this layout.
import worker.src.sandbox.sandbox as sandbox_mod  # noqa: F401 (ensures the module resolves in this layout)

SAVED_WORKER = sys.modules.get("worker")
sys.modules["worker.src"] = importlib.import_module("worker.src")
sys.modules["worker"] = SAVED_WORKER


def _train_1k_in_child(
    queue, threads: int, steps: int, seed: int, mem_mb: int, cpu_sec: int
) -> None:
    """Runs in a forked child; applies Linux rlimits like the real sandbox."""
    # MANDATORY: strict rlimits; a child that cannot apply them must fail.
    try:
        apply_strict_rlimits()
    except Exception as e:
        queue.put({"status": "error", "error": f"rlimit setup failed: {e}"})
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.set_num_threads(threads)

        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(10, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
        )  # 10*30+30 + 30 + 30*30+30 + 30 + 30*1+1 = 1,241 params
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

        grads = [p.grad.detach().flatten().tolist() for p in model.parameters()]
        flat = [v for g in grads for v in g]
        queue.put(
            {
                "status": "ok",
                "n_params": n_params,
                "wall_s": wall,
                "steps_per_s": steps / wall,
                "ms_per_step": 1000.0 * wall / steps,
                "final_loss": losses[-1],
                "first5_losses": [round(v, 6) for v in losses[:5]],
                "grad_len": len(flat),
                "grad_sum": sum(flat),
                "grad_first3": flat[:3],
            }
        )
    except Exception as e:  # noqa: BLE001
        queue.put({"status": "error", "error": f"{type(e).__name__}: {e}"})


def _median(values):
    return statistics.median(values) if values else 0.0


def run_sandboxed(threads: int, steps: int, seed: int, mem_mb: int = 4096) -> dict:
    """One training run inside a fresh, rlimit-capped child process.

    Mirrors worker/src/sandbox/sandbox.py:Sandbox._run_subprocess
    (fork + resource.setrlimit + result via pipe), without the pickle
    layer so numbers reflect the training loop itself.
    """
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    t0 = time.perf_counter()
    proc = ctx.Process(
        target=_train_1k_in_child,
        args=(queue, threads, steps, seed, mem_mb, 300),
    )
    proc.start()
    proc.join(180)
    spawn_s = time.perf_counter() - t0
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {"status": "timeout", "spawn_s": spawn_s}
    result = (
        queue.get()
        if not queue.empty()
        else {"status": "crash", "error": f"exitcode={proc.exitcode}"}
    )
    result["spawn_join_s"] = spawn_s
    result["threads"] = threads
    return result


def run_config(name: str, n_sandboxes: int, total_threads: int, steps: int) -> dict:
    """Split total_threads across n_sandboxes and train concurrently."""
    threads_each = max(1, total_threads // n_sandboxes)
    ctx = mp.get_context("fork")
    queue = ctx.Queue()
    t0 = time.perf_counter()
    procs = []
    for i in range(n_sandboxes):
        p = ctx.Process(
            target=_train_1k_in_child,
            args=(queue, threads_each, steps, 42 + i, 4096, 300),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(240)
    wall = time.perf_counter() - t0
    results = []
    while not queue.empty():
        results.append(queue.get())
    ok = [r for r in results if r.get("status") == "ok"]
    return {
        "config": name,
        "n_sandboxes": n_sandboxes,
        "threads_each": threads_each,
        "wall_s": round(wall, 4),
        "all_ok": len(ok) == n_sandboxes,
        "results": [
            {
                "steps_per_s": round(r["steps_per_s"], 1),
                "ms_per_step": round(r["ms_per_step"], 3),
                "final_loss": round(r["final_loss"], 6),
                "n_params": r["n_params"],
            }
            for r in results
        ],
    }


def bench_matmul(size: int, iters: int, dtype: str) -> float:
    import torch

    a = torch.rand(size, size)
    b = torch.rand(size, size)
    if dtype == "fp32":
        a, b = a.float(), b.float()
    elif dtype == "fp64":
        a, b = a.double(), b.double()
    # warmup
    for _ in range(3):
        _ = a @ b
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    return (time.perf_counter() - t0) / iters


def micro_workloads() -> dict:
    import numpy as np
    import torch

    torch.set_num_threads(1)

    out = {}

    # matmul 256x256 fp32, single-thread
    out["matmul_256_fp32_ms"] = round(bench_matmul(256, 200, "fp32") * 1000, 4)

    # argmax over 1M floats, pure python loop vs torch
    n = 1_000_000
    data = [float(i % 1000) for i in range(n)]
    t0 = time.perf_counter()
    best_v = data[0]
    for _, v in enumerate(data):
        if v > best_v:
            best_v = v
    out["argmax_py_1m_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    arr = np.array(data)
    t0 = time.perf_counter()
    _ = int(np.argmax(arr))
    out["argmax_np_1m_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    # serialization: dict of 100 float lists x 1000
    payload = {f"layer_{i}": [float(j) * 0.001 for j in range(1000)] for i in range(100)}
    t0 = time.perf_counter()
    s = json.dumps(payload)
    out["json_dumps_100x1000_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    t0 = time.perf_counter()
    _ = json.loads(s)
    out["json_parse_100x1000_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    # tensor -> python list -> JSON string (the worker gradient path)
    t = torch.rand(100_000)
    t0 = time.perf_counter()
    lst = t.tolist()
    s = json.dumps(lst)
    out["grad_to_json_100k_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    t0 = time.perf_counter()
    _ = json.loads(s)
    out["json_to_grad_100k_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    arr = t.numpy()
    t0 = time.perf_counter()
    _ = arr.tobytes()
    out["grad_to_bytes_100k_ms"] = round((time.perf_counter() - t0) * 1000, 3)

    # allocator churn: 10k small tensor allocations
    t0 = time.perf_counter()
    for _ in range(10_000):
        _ = torch.rand(16)
    out["alloc_10k_small_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return out


def train_scaling() -> dict:
    """Same regression task as bench_tensor.py at several model sizes."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.set_num_threads(1)
    results = {}
    for name, hidden in [("1param", 0), ("1k", 30), ("10k", 90), ("100k", 290)]:
        if hidden == 0:
            w = nn.Parameter(torch.ones(1))
            params = [w]
            forward = lambda x, _w=w: _w * x  # noqa: E731
            n_params = 1
        else:
            model = nn.Sequential(
                nn.Linear(1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
            params = list(model.parameters())
            n_params = sum(p.numel() for p in params)
            forward = model
        opt = optim.AdamW(params, lr=1e-3)
        crit = nn.MSELoss()
        # warmup
        for _ in range(20):
            x = torch.rand(1, 1)
            y = torch.sin(3.14159 * x) + 0.5 * torch.sin(3 * 3.14159 * x)
            opt.zero_grad()
            crit(forward(x), y).backward()
            opt.step()
        t0 = time.perf_counter()
        n = 300
        for _ in range(n):
            x = torch.rand(1, 1)
            y = torch.sin(3.14159 * x) + 0.5 * torch.sin(3 * 3.14159 * x)
            opt.zero_grad()
            crit(forward(x), y).backward()
            opt.step()
        dt = time.perf_counter() - t0
        results[name] = {
            "params": n_params,
            "steps_per_s": round(n / dt, 1),
            "ms_per_step": round(1000 * dt / n, 3),
        }
    return results


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer steps/iters")
    args = ap.parse_args()

    steps = 200 if args.quick else 1000
    print(f"== DistribAI baseline (Python/PyTorch) -- quick={args.quick} ==")

    preflight()
    out = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname": platform.node(),
            "resource_limits": {
                **SANDBOX_RLIMITS,
                "min_free_ram_mb": MIN_FREE_RAM_MB,
                "available_ram_mb_at_start": _available_ram_mb(),
            },
            "python": sys.version.split()[0],
            "torch_version": None,
            "cuda_available": False,
            "cpu": None,
            "cores": os.cpu_count(),
        }
    }

    import torch

    out["meta"]["torch_version"] = torch.__version__
    out["meta"]["cuda_available"] = torch.cuda.is_available()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    out["meta"]["cpu"] = line.split(":")[1].strip()
                    break
    except OSError:
        pass

    print("-- micro workloads --")
    out["micro"] = micro_workloads()
    for k, v in out["micro"].items():
        print(f"  {k:30s} {v}")

    print("-- training scaling (single proc, 1 thread) --")
    out["train_scaling"] = train_scaling()
    for k, v in out["train_scaling"].items():
        print(f"  {k:8s} {v['params']:>7d} params  {v['steps_per_s']:>8.1f} steps/s")

    print("-- sandboxed 1K-param training (per-sandbox, best of 3) --")
    out["sandbox_1k"] = {}
    reps = 3
    for threads in (1, 4):
        runs = [run_sandboxed(threads, steps, seed=42 + i, mem_mb=4096) for i in range(reps)]
        oks = [r for r in runs if r.get("status") == "ok"]
        key = f"{threads}t"
        best = max(oks, key=lambda r: r["steps_per_s"]) if oks else runs[0]
        out["sandbox_1k"][key] = {
            "best": best,
            "all_runs": [
                {
                    k: r.get(k)
                    for k in ("status", "steps_per_s", "ms_per_step", "final_loss", "spawn_join_s")
                }
                for r in runs
            ],
        }
        r = best
        if r.get("status") == "ok":
            print(
                f"  {threads}t: {r['steps_per_s']:.1f} steps/s "
                f"({r['ms_per_step']:.2f} ms/step) loss={r['final_loss']:.4f}"
            )
        else:
            print(f"  {threads}t: {r}")

    print("-- config: 1 sandbox (100%) vs 2 sandboxes (50% each), best of 3 --")
    out["configs"] = {}
    for name, n, tot in [
        ("sandbox1x", 1, 4),
        ("sandbox2x", 2, 4),
    ]:
        runs = [run_config(name, n, tot, steps) for _ in range(reps)]
        good = [r for r in runs if r["all_ok"]]
        best = min(good, key=lambda r: r["wall_s"]) if good else runs[0]
        r = dict(best)
        r["all_runs_wall_s"] = [x["wall_s"] for x in runs]
        out["configs"][name] = r
        print(
            f"  {name}: wall={r['wall_s']}s ok={r['all_ok']} "
            f"per-sbx={[x['steps_per_s'] for x in r['results']]}"
        )

    print("-- end-to-end gradient handoff cycle --")
    import torch as th

    model = th.nn.Sequential(
        th.nn.Linear(10, 30), th.nn.ReLU(), th.nn.Linear(30, 30), th.nn.ReLU(), th.nn.Linear(30, 1)
    )
    opt = th.optim.AdamW(model.parameters(), lr=0.01)
    cycles = 50 if args.quick else 100
    t0 = time.perf_counter()
    for _ in range(cycles):
        x = th.rand(64, 10)
        y = th.rand(64, 1)
        opt.zero_grad()
        th.nn.functional.mse_loss(model(x), y).backward()
        grads = [p.grad.flatten().tolist() for p in model.parameters()]
        blob = json.dumps({"grads": grads})
        _ = json.loads(blob)
    dt = time.perf_counter() - t0
    out["e2e_cycle"] = {
        "cycles": cycles,
        "wall_s": round(dt, 4),
        "ms_per_cycle": round(1000 * dt / cycles, 3),
        "steps_per_s": round(cycles / dt, 1),
    }
    print(
        f"  {cycles} cycles: {out['e2e_cycle']['ms_per_cycle']} ms/cycle "
        f"({out['e2e_cycle']['steps_per_s']} cycles/s)"
    )

    dest = REPO / "runtime" / "baselines"
    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = dest / f"baseline_python_{stamp}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
