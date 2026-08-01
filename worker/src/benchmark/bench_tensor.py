"""
DistribAI Benchmark: Tensor Compute
====================================
Trains tiny-to-medium neural networks and measures training throughput.
Model sizes: 1 → 10 → 100 → 1 K → 10 K → 100 K → 1 M parameters.
Each model learns to predict the next value in a mathematical sequence
(Fibonacci-like recurrence), giving a concrete, reproducible task.
Settings per the benchmark spec:
  batch_size = 1,  gradient_accumulation = 1
Device: GPU (CUDA / MPS) if available, else CPU.
Score calibration
-----------------
SCORE_FLOOR_SPS   (env) → 0   score  (steps/s for 1M-param model, default 1)
SCORE_CEIL_SPS    (env) → 100 score  (steps/s for 1M-param model, default 50 000)
TENSOR_STEPS      (env) steps per model size (default 300)
TENSOR_WARMUP     (env) warmup steps discarded (default 30)
"""

import json
import math
import os
import time

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    optim = None
    _HAS_TORCH = False
_HAS_PSUTIL = False
_FLOOR = float(os.environ.get("SCORE_FLOOR_SPS", 1.0))
_CEIL = float(os.environ.get("SCORE_CEIL_SPS", 50_000.0))
_STEPS = int(os.environ.get("TENSOR_STEPS", 60))
_WARMUP = int(os.environ.get("TENSOR_WARMUP", 6))
_PARAM_TARGETS = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def _count_params(model: "nn.Module") -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_model(target_params: int) -> "nn.Module":
    """
    Return the smallest MLP/transformer whose parameter count >= target_params.
    All models: scalar → scalar (sequence prediction task).
    """
    import torch.nn as nn

    if target_params <= 1:

        class OneParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(1))

            def forward(self, x):
                return self.w * x

        return OneParam()
    if target_params <= 16:
        d = max(2, target_params // 2)
        return nn.Sequential(nn.Linear(1, d), nn.Tanh(), nn.Linear(d, 1))
    if target_params <= 200:
        d = max(4, int(math.sqrt(target_params / 3)))
        while 3 * d * d + 2 * d < target_params:
            d += 1
        return nn.Sequential(
            nn.Linear(1, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1)
        )
    if target_params <= 2_000:
        d = 8
        while True:
            m = nn.Sequential(
                nn.Linear(1, d),
                nn.ReLU(),
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, 1),
            )
            if _count_params(m) >= target_params:
                return m
            d = int(d * 1.5)
    if target_params <= 20_000:
        d = 32
        while True:
            m = nn.Sequential(
                nn.Linear(1, d),
                nn.GELU(),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, 1),
            )
            if _count_params(m) >= target_params:
                return m
            d = int(d * 1.4)
    if target_params <= 200_000:

        class TinyTransformer(nn.Module):
            def __init__(self, d_model, n_head, n_layers, seq_len=8):
                super().__init__()
                self.embed = nn.Linear(1, d_model)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    dim_feedforward=d_model * 2,
                    dropout=0.0,
                    batch_first=True,
                )
                self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
                self.head = nn.Linear(d_model, 1)
                self.seq_len = seq_len

            def forward(self, x):
                x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
                x = self.embed(x)
                x = self.enc(x)
                return self.head(x[:, -1, :])

        for n_layers in [1, 2, 3]:
            for d_model in [16, 32, 48, 64, 96, 128, 192, 256]:
                for n_head in [1, 2, 4]:
                    if d_model % n_head != 0:
                        continue
                    try:
                        m = TinyTransformer(d_model, n_head, n_layers)
                        if _count_params(m) >= target_params:
                            return m
                    except (RuntimeError, ValueError):
                        continue
        return TinyTransformer(128, 4, 2)

    class MedTransformer(nn.Module):
        def __init__(self, d_model=256, n_head=8, n_layers=4, seq_len=16):
            super().__init__()
            self.embed = nn.Linear(1, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=d_model * 4,
                dropout=0.0,
                batch_first=True,
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.head = nn.Linear(d_model, 1)
            self.seq_len = seq_len

        def forward(self, x):
            x = x.unsqueeze(1).expand(-1, self.seq_len, -1)
            x = self.embed(x)
            x = self.enc(x)
            return self.head(x[:, -1, :])

    for n_layers in [2, 3, 4, 5, 6]:
        for d_model in [128, 192, 256, 320, 384, 512]:
            for n_head in [4, 8]:
                if d_model % n_head != 0:
                    continue
                try:
                    m = MedTransformer(d_model, n_head, n_layers)
                    if _count_params(m) >= target_params:
                        return m
                except (RuntimeError, ValueError):
                    continue
    return MedTransformer()


def _make_batch(device, step: int) -> tuple:
    """
    Task: given x ∈ [0,1], predict f(x) = sin(π·x) + 0.5·sin(3π·x).
    A well-defined non-trivial regression — every model can make progress.
    batch_size=1, grad_accum=1 as specified.
    """
    x = torch.rand(1, 1, device=device)
    y = torch.sin(math.pi * x) + 0.5 * torch.sin(3 * math.pi * x)
    return x, y


def _mean(values: list[float]) -> float:
    """Calculate mean with fallback if numpy not available."""
    if not values:
        return 0.0
    if _HAS_NUMPY:
        return float(np.mean(values))
    return sum(values) / len(values)


def _detect_throttle(throughputs: list[float]) -> bool:
    if len(throughputs) < 60:
        return False
    early = _mean(throughputs[: len(throughputs) // 3])
    late = _mean(throughputs[len(throughputs) * 2 // 3 :])
    return early > 0 and late / early < 0.82


def _train_model(model, device, target: int, steps: int, warmup: int) -> dict:
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    step_times: list[float] = []
    losses: list[float] = []
    for step in range(steps + warmup):
        x, y = _make_batch(device, step)
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        pred = model(x)
        loss = crit(pred, y)
        loss.backward()
        opt.step()
        if hasattr(torch.cuda, "synchronize") and device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= warmup:
            step_times.append(dt)
            losses.append(float(loss.detach()))
    sps = 1.0 / _mean(step_times)
    loss_mean = _mean(losses[-min(50, len(losses)) :])
    throttled = _detect_throttle(step_times)
    return {
        "steps_per_sec": sps,
        "ms_per_step": 1000.0 / sps,
        "final_loss": loss_mean,
        "throttled": throttled,
    }


def main():
    if not _HAS_TORCH:
        emit(
            {
                "type": "error",
                "test": "tensor",
                "message": "PyTorch not installed — skipping tensor benchmark.",
            }
        )
        return None
    emit(
        {
            "type": "benchmark_group_start",
            "name": "tensor",
            "display": "Tensor Compute",
            "message": (
                "Training models from 1 → 1 M parameters. "
                "Task: regress f(x)=sin(πx)+½sin(3πx). "
                "batch=1, grad_accum=1."
            ),
        }
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    emit(
        {
            "type": "progress",
            "test": "tensor",
            "pct": 2,
            "message": f"Device: {device} | {_STEPS} steps per model size (+ {_WARMUP} warmup)",
        }
    )
    results_per_size: list[dict] = []
    any_throttled = False
    n = len(_PARAM_TARGETS)
    for idx, target in enumerate(_PARAM_TARGETS):
        model = _build_model(target)
        actual = _count_params(model)
        label = f"{actual:,} params"
        pct_start = int((idx / n) * 90)
        emit(
            {
                "type": "progress",
                "test": "tensor",
                "pct": pct_start,
                "message": f"Training {label} model… ({idx + 1}/{n})",
                "model_idx": idx,
                "param_count": actual,
            }
        )
        try:
            stats = _train_model(model, device, target, _STEPS, _WARMUP)
        except Exception as e:
            emit({"type": "error", "test": "tensor", "message": f"Model {label} failed: {e}"})
            continue
        if stats["throttled"]:
            any_throttled = True
            emit(
                {
                    "type": "thermal_warning",
                    "test": "tensor",
                    "message": f"Throughput drop detected while training {label} model.",
                }
            )
        emit(
            {
                "type": "progress",
                "test": "tensor",
                "pct": int(((idx + 0.9) / n) * 90),
                "message": (
                    f"{label}: {stats['steps_per_sec']:.1f} steps/s  "
                    f"({stats['ms_per_step']:.2f} ms/step)  "
                    f"loss={stats['final_loss']:.4f}"
                ),
                "param_count": actual,
                "steps_per_sec": round(stats["steps_per_sec"], 2),
                "ms_per_step": round(stats["ms_per_step"], 3),
            }
        )
        results_per_size.append(
            {
                "target_params": target,
                "actual_params": actual,
                "steps_per_sec": round(stats["steps_per_sec"], 2),
                "ms_per_step": round(stats["ms_per_step"], 3),
                "final_loss": round(stats["final_loss"], 6),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    emit(
        {
            "type": "progress",
            "test": "tensor",
            "pct": 95,
            "message": "All model sizes complete — computing score…",
        }
    )
    primary_sps = 0.0
    if results_per_size:
        primary_sps = results_per_size[-1]["steps_per_sec"]
    score = log_score(primary_sps, _FLOOR, _CEIL)
    result = {
        "type": "result",
        "test": "tensor",
        "device": str(device),
        "gpu_used": device.type in ("cuda", "mps"),
        "sizes": results_per_size,
        "largest_sps": round(primary_sps, 2),
        "score": round(score, 1),
        "thermal_throttled": any_throttled,
    }
    emit(result)
    return result


if __name__ == "__main__":
    main()
