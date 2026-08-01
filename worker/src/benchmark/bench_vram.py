"""
DistribAI Benchmark: VRAM Capacity
===================================
Determines how large a model can be trained (full forward + backward pass,
batch_size=1) at each available dtype: fp32, fp16, bf16.
Also measures the raw maximum allocatable tensor.
Strategy
--------
Rather than an open-ended bisection (slow on large-VRAM GPUs), this benchmark
tests a fixed sequence of model sizes in MB and records the largest that fits.
The sequence covers 128 MB → up to 80 % of total VRAM in doubling steps, capped
at a reasonable upper bound so the test completes in reasonable time.
Score calibration
-----------------
SCORE_FLOOR_GB  (env) → 0   score  (default 0.5 GB usable VRAM)
SCORE_CEIL_GB   (env) → 100 score  (default 128 GB usable VRAM)
"""

import json
import math
import os
from typing import Any

_FLOOR = float(os.environ.get("SCORE_FLOOR_GB", 0.5))
_CEIL = float(os.environ.get("SCORE_CEIL_GB", 128.0))

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
    _HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    _HAS_ANY_GPU = _HAS_CUDA or _HAS_MPS
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH = _HAS_CUDA = _HAS_MPS = _HAS_ANY_GPU = False


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def _build_flat_model(target_mb: float, dtype: Any) -> Any:
    """
    Single-layer MLP sized to approximately `target_mb` MB of parameters.
    bytes_per_param: 4 for fp32, 2 for fp16/bf16.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required for VRAM benchmark")
    bpp = 4 if dtype == torch.float32 else 2
    n_par = int(target_mb * 2**20 / bpp)
    d = max(2, int(math.sqrt(max(n_par, 4))))
    return nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))


def _try_train(device, dtype: Any, target_mb: float) -> bool:
    """
    Try to build and run one forward+backward pass at the given dtype and size.
    Returns True on success, False on OOM or any error.
    """
    crit = nn.MSELoss()
    try:
        if getattr(device, "type", None) is None:
            try:
                device = torch.device(device)
            except RuntimeError:
                return False

        # For MPS, some dtypes may not be supported
        if device.type == "mps" and dtype == torch.bfloat16:
            return False  # Skip bf16 on MPS for now

        model = _build_flat_model(target_mb, dtype).to(device=device, dtype=dtype)
        d_in = next(model.parameters()).shape[-1]
        x = torch.randn(1, d_in, device=device, dtype=dtype)
        y = torch.randn(1, 1, device=device, dtype=dtype)
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        del model, x, y, out, loss

        # Clear cache based on device type
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        try:
            dev = device if getattr(device, "type", None) is not None else torch.device(device)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            elif dev.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass
        return False
    except (TypeError, ValueError):
        try:
            dev = device if getattr(device, "type", None) is not None else torch.device(device)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            elif dev.type == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass
        return False


def _max_raw_alloc_gb(device) -> float:
    if not hasattr(device, "type"):
        try:
            device = torch.device(device)
        except RuntimeError:
            return 0.0

    # Get total memory based on device type
    if device.type == "cuda":
        try:
            total_gb = torch.cuda.get_device_properties(device).total_memory / 2**30
        except RuntimeError:
            return 0.0
    elif device.type == "mps":
        import psutil

        total_memory = psutil.virtual_memory().total / 2**30
        total_gb = total_memory * 0.4  # Estimate shared memory for MPS
    else:  # CPU
        import psutil

        total_gb = psutil.virtual_memory().available / 2**30 * 0.8

    lo, hi = 0.01, total_gb * 0.95
    best = 0.0
    for _ in range(22):
        mid = (lo + hi) / 2
        try:
            t = torch.empty(int(mid * 2**30 / 4), dtype=torch.float32, device=device)
            del t
            # Clear cache based on device type
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            best = mid
            lo = mid
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            # Clear cache based on device type
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()
            hi = mid
        if hi - lo < 0.05:
            break
    return best


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "vram",
            "display": "VRAM Capacity",
            "message": "Probing maximum raw allocation and trainable model size per dtype…",
        }
    )
    has_gpu_backend = _HAS_CUDA or _HAS_MPS
    if not _HAS_TORCH or not has_gpu_backend:
        reason_parts = []
        if not _HAS_TORCH:
            reason_parts.append("PyTorch not available")
        if not _HAS_CUDA and not _HAS_MPS:
            reason_parts.append("no GPU backends available")
        elif not _HAS_CUDA:
            reason_parts.append("CUDA not available")
        elif not _HAS_MPS:
            reason_parts.append("MPS not available")

        emit(
            {
                "type": "skip",
                "test": "vram",
                "reason": ", ".join(reason_parts) + " — VRAM benchmark skipped.",
            }
        )
        emit(
            {
                "type": "result",
                "test": "vram",
                "score": 0.0,
                "thermal_throttled": False,
                "max_usable_gb": 0.0,
                "skip": True,
            }
        )
        return None

    # Choose the best available GPU backend
    if _HAS_CUDA:
        device = torch.device("cuda")
        backend_name = "CUDA"
        total_vram = torch.cuda.get_device_properties(device).total_memory / 2**30
    elif _HAS_MPS:
        device = torch.device("mps")
        backend_name = "Apple Metal (MPS)"
        # For MPS, we need to estimate memory since there's no direct API
        import psutil

        total_memory = psutil.virtual_memory().total / 2**30
        total_vram = total_memory * 0.4  # Estimate shared memory for MPS
    else:
        # Fallback to CPU for basic testing (will be very limited)
        device = torch.device("cpu")
        backend_name = "CPU"
        import psutil

        total_vram = psutil.virtual_memory().available / 2**30 * 0.8
    emit(
        {
            "type": "progress",
            "test": "vram",
            "pct": 5,
            "message": f"Backend: {backend_name} — Total Memory: {total_vram:.1f} GB — finding max raw allocation…",
        }
    )
    max_raw = _max_raw_alloc_gb(device)
    emit(
        {
            "type": "progress",
            "test": "vram",
            "pct": 18,
            "message": f"Max contiguous allocation: {max_raw:.2f} GB — testing trainable model sizes…",
        }
    )
    cap_mb = total_vram * 0.40 * 1024  # Test up to 40% of total VRAM for safety
    test_mbs: list[float] = []
    s = 128.0
    while s <= cap_mb:
        test_mbs.append(s)
        s *= 2
    dtype_results: dict[str, float] = {}
    n_dtypes = 3
    dtype_pct_base = 18
    for d_idx, (dtype, label) in enumerate(
        [
            (torch.float32, "fp32"),
            (torch.float16, "fp16"),
        ]
    ):
        best_mb = 0.0
        for i, mb in enumerate(test_mbs):
            pct = dtype_pct_base + int(
                (d_idx / n_dtypes + (i / max(1, len(test_mbs))) / n_dtypes) * 70
            )
            emit(
                {
                    "type": "progress",
                    "test": "vram",
                    "pct": min(88, pct),
                    "message": f"Testing {label}: {mb / 1024:.2f} GB model…",
                }
            )
            if _try_train(device, dtype, mb):
                best_mb = mb
            else:
                break
        dtype_results[label] = best_mb
        emit(
            {
                "type": "progress",
                "test": "vram",
                "pct": dtype_pct_base + int((d_idx + 1) / n_dtypes * 70),
                "message": f"{label} max trainable: {best_mb / 1024:.2f} GB model params",
            }
        )
    # Check bf16 support based on backend
    bf16_supported = False
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        bf16_supported = True
    elif device.type == "mps" and hasattr(torch, "bfloat16"):
        # MPS bf16 support is limited, check if we can try it
        bf16_supported = True

    if bf16_supported:
        best_mb = 0.0
        for i, mb in enumerate(test_mbs):
            emit(
                {
                    "type": "progress",
                    "test": "vram",
                    "pct": 88 + int(i / max(1, len(test_mbs)) * 8),
                    "message": f"Testing bf16: {mb / 1024:.2f} GB model…",
                }
            )
            if _try_train(device, torch.bfloat16, mb):
                best_mb = mb
            else:
                break
        dtype_results["bf16"] = best_mb
    emit({"type": "progress", "test": "vram", "pct": 97, "message": "Computing score…"})
    best_dtype = max(dtype_results, key=dtype_results.get) if dtype_results else "none"
    best_mb = dtype_results.get(best_dtype, 0)
    score = log_score(max_raw, _FLOOR, _CEIL)
    result = {
        "type": "result",
        "test": "vram",
        "backend": backend_name,
        "device_type": device.type,
        "total_vram_gb": round(total_vram, 2),
        "max_raw_alloc_gb": round(max_raw, 3),
        "max_usable_gb": round(max_raw, 3),
        "dtype_max_trainable_gb": {k: round(v / 1024, 3) for k, v in dtype_results.items()},
        "best_dtype": best_dtype,
        "best_dtype_max_gb": round(best_mb / 1024, 3) if best_mb else 0,
        "bf16_supported": bf16_supported,
        "score": round(score, 1),
        "thermal_throttled": False,
    }
    emit(result)
    return result


if __name__ == "__main__":
    main()
