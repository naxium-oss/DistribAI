"""
DistribAI Benchmark: Memory
==========================
Tests system RAM performance:
  1. Sequential write bandwidth  (GB/s)
  2. Sequential read  bandwidth  (GB/s)
  3. Random-access latency       (ns, pointer-chasing)
  4. Copy bandwidth              (GB/s)
Score calibration
-----------------
SCORE_FLOOR_GBS  (env)  → 0   score  (default 0.5 GB/s)
SCORE_CEIL_GBS   (env)  → 100 score  (default 300 GB/s)
MEM_DURATION_S   (env)  test time per sub-test (default 15s)
"""

import json
import math
import os
import time

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
_FLOOR = float(os.environ.get("SCORE_FLOOR_GBS", 0.5))
_CEIL = float(os.environ.get("SCORE_CEIL_GBS", 300.0))
_DUR = float(os.environ.get("MEM_DURATION_S", 3.0))
_MAX_CHUNK_BYTES = int(os.environ.get("BENCH_MEM_MAX_CHUNK_BYTES", str(2 * 2**30)))


def _choose_array_bytes() -> int:
    if _HAS_PSUTIL:
        free = psutil.virtual_memory().available
        target = int(min(free * 0.35, _MAX_CHUNK_BYTES))
    else:
        target = 512 * 2**20
    return max(target, 256 * 2**20)


def _probe_allocatable_bytes(preferred: int) -> int:
    """Shrink preferred size until a probe ndarray allocates (handles fragmentation/OOM)."""
    if not _HAS_NUMPY:
        return preferred
    minimum = 128 * 2**20
    n = min(preferred, _MAX_CHUNK_BYTES)
    while n >= minimum:
        try:
            probe = np.empty(n // 8, dtype=np.float64)
            del probe
            return n
        except MemoryError:
            n //= 2
        except Exception as exc:
            if "Unable to allocate" in str(exc):
                n //= 2
                continue
            raise
    return minimum


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float | None, floor: float, ceil: float) -> float:
    if value is None or value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def bench_sequential_write(arr_bytes: int, duration: float) -> float | None:
    if not _HAS_NUMPY:
        return None
    chunk = np.empty(arr_bytes // 8, dtype=np.float64)
    n_iter, total_bytes, t0 = 0, 0, time.perf_counter()
    while time.perf_counter() - t0 < duration:
        chunk[:] = 1.23456
        total_bytes += chunk.nbytes
        n_iter += 1
    return total_bytes / (time.perf_counter() - t0) / 1e9


def bench_sequential_read(arr_bytes: int, duration: float) -> float | None:
    if not _HAS_NUMPY:
        return None
    chunk = np.ones(arr_bytes // 8, dtype=np.float64)
    n_iter, total_bytes, t0 = 0, 0, time.perf_counter()
    sink = 0.0
    while time.perf_counter() - t0 < duration:
        sink += float(chunk.sum())
        total_bytes += chunk.nbytes
        n_iter += 1
    return total_bytes / (time.perf_counter() - t0) / 1e9


def bench_copy(arr_bytes: int, duration: float) -> float | None:
    if not _HAS_NUMPY:
        return None
    src = np.ones(arr_bytes // 16, dtype=np.float64)
    dst = np.empty_like(src)
    total_bytes, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < duration:
        np.copyto(dst, src)
        total_bytes += src.nbytes + dst.nbytes
    return total_bytes / (time.perf_counter() - t0) / 1e9


def bench_latency(n_pointers: int = 8 * 2**20, n_walks: int = 5_000_000) -> float | None:
    """
    Pointer-chasing random-access latency (ns).
    Constructs a random linked-list in a large int32 array, then follows it.
    """
    if not _HAS_NUMPY:
        return None
    rng = np.random.default_rng()
    chain = rng.permutation(n_pointers).astype(np.int32)
    t0 = time.perf_counter()
    idx = 0
    steps = 0
    limit = min(n_walks, n_pointers * 3)
    while steps < limit:
        idx = chain[idx]
        steps += 1
    dt_ns = (time.perf_counter() - t0) / steps * 1e9
    _ = idx
    return dt_ns


def _fmt(value: float | None, fmt: str, default: str = "N/A") -> str:
    """Format value with fallback for None."""
    if value is None:
        return default
    return f"{value:{fmt}}"


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "memory",
            "display": "Memory",
            "message": "Measuring RAM sequential bandwidth and random-access latency…",
        }
    )
    arr_bytes = _probe_allocatable_bytes(_choose_array_bytes())
    arr_mb = arr_bytes / 2**20
    emit(
        {
            "type": "progress",
            "test": "memory",
            "pct": 5,
            "message": f"Array size: {arr_mb:.0f} MB — testing sequential write…",
        }
    )
    write_gbs = bench_sequential_write(arr_bytes, _DUR) if _HAS_NUMPY else None
    emit(
        {
            "type": "progress",
            "test": "memory",
            "pct": 30,
            "message": f"Write: {_fmt(write_gbs, '.2f')} GB/s — testing sequential read…",
            "write_gbs": round(write_gbs, 3) if write_gbs else None,
        }
    )
    read_gbs = bench_sequential_read(arr_bytes, _DUR) if _HAS_NUMPY else None
    emit(
        {
            "type": "progress",
            "test": "memory",
            "pct": 55,
            "message": f"Read: {_fmt(read_gbs, '.2f')} GB/s — testing copy bandwidth…",
            "read_gbs": round(read_gbs, 3) if read_gbs else None,
        }
    )
    copy_gbs = bench_copy(arr_bytes, _DUR) if _HAS_NUMPY else None
    emit(
        {
            "type": "progress",
            "test": "memory",
            "pct": 75,
            "message": f"Copy: {_fmt(copy_gbs, '.2f')} GB/s — measuring random-access latency…",
            "copy_gbs": round(copy_gbs, 3) if copy_gbs else None,
        }
    )
    lat_ns = bench_latency() if _HAS_NUMPY else None
    emit(
        {
            "type": "progress",
            "test": "memory",
            "pct": 95,
            "message": f"Latency: {_fmt(lat_ns, '.1f')} ns — computing score…",
        }
    )
    score = log_score(read_gbs, _FLOOR, _CEIL)
    result = {
        "type": "result",
        "test": "memory",
        "write_gbs": round(write_gbs, 3) if write_gbs else None,
        "read_gbs": round(read_gbs, 3) if read_gbs else None,
        "copy_gbs": round(copy_gbs, 3) if copy_gbs else None,
        "latency_ns": round(lat_ns, 1) if lat_ns else None,
        "array_size_mb": round(arr_mb, 0),
        "score": round(score, 1),
        "thermal_throttled": False,
    }
    emit(result)
    return result


if __name__ == "__main__":
    main()
