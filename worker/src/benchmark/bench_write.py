"""
DistribAI Benchmark: Disk Write / I/O
======================================
Tests:
  1. Sequential write  (MB/s)
  2. Sequential read   (MB/s)
  3. Random 4 KB write (IOPS)
  4. Random 4 KB read  (IOPS)
All tests use a temporary directory; files are cleaned up afterwards.
Score calibration
-----------------
SCORE_FLOOR_MBS  (env) → 0   score (default 10 MB/s)
SCORE_CEIL_MBS   (env) → 100 score (default 15 000 MB/s)
WRITE_DURATION_S (env) seconds per sub-test (default 15 s)
"""

import json
import math
import os
import tempfile
import time

_FLOOR = float(os.environ.get("SCORE_FLOOR_MBS", 10.0))
_CEIL = float(os.environ.get("SCORE_CEIL_MBS", 15_000.0))
_DUR = float(os.environ.get("WRITE_DURATION_S", 3.0))
_SEQ_BLOCK = 4 * 2**20
_RAND_BLOCK = 4 * 2**10


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def _seq_write(path: str, block: bytes, duration: float) -> float:
    total = 0
    t0 = time.perf_counter()
    with open(path, "wb", buffering=0) as f:
        while time.perf_counter() - t0 < duration:
            f.write(block)
            total += len(block)
    return total / (time.perf_counter() - t0) / 2**20


def _seq_read(path: str, buf_size: int, duration: float) -> float:
    total = 0
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        while time.perf_counter() - t0 < duration:
            chunk = f.read(buf_size)
            if not chunk:
                f.seek(0)
                continue
            total += len(chunk)
    return total / (time.perf_counter() - t0) / 2**20


def _rand_write(path: str, file_size: int, block_size: int, duration: float) -> float:
    block = os.urandom(block_size)
    n_pos = file_size // block_size
    iops_count = 0
    t0 = time.perf_counter()
    with open(path, "r+b", buffering=0) as f:
        while time.perf_counter() - t0 < duration:
            pos = (hash(str(time.perf_counter_ns())) % n_pos) * block_size
            f.seek(pos)
            f.write(block)
            iops_count += 1
    return iops_count / (time.perf_counter() - t0)


def _rand_read(path: str, file_size: int, block_size: int, duration: float) -> float:
    n_pos = file_size // block_size
    iops_count = 0
    t0 = time.perf_counter()
    with open(path, "rb", buffering=0) as f:
        while time.perf_counter() - t0 < duration:
            pos = (hash(str(time.perf_counter_ns())) % n_pos) * block_size
            f.seek(pos)
            _ = f.read(block_size)
            iops_count += 1
    return iops_count / (time.perf_counter() - t0)


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "write",
            "display": "Disk I/O",
            "message": "Measuring sequential and random disk throughput…",
        }
    )
    with tempfile.TemporaryDirectory(prefix="distribai_bench_") as tmpdir:
        seq_path = os.path.join(tmpdir, "seq.bin")
        rand_path = os.path.join(tmpdir, "rand.bin")
        seq_block = os.urandom(_SEQ_BLOCK)
        emit({"type": "progress", "test": "write", "pct": 5, "message": "Sequential write…"})
        seq_w = _seq_write(seq_path, seq_block, _DUR)
        emit(
            {
                "type": "progress",
                "test": "write",
                "pct": 30,
                "message": f"Seq write: {seq_w:.0f} MB/s — sequential read…",
                "seq_write_mbs": round(seq_w, 1),
            }
        )
        seq_r = _seq_read(seq_path, _SEQ_BLOCK, _DUR)
        emit(
            {
                "type": "progress",
                "test": "write",
                "pct": 55,
                "message": f"Seq read: {seq_r:.0f} MB/s — random write…",
                "seq_read_mbs": round(seq_r, 1),
            }
        )
        rand_file_size = min(256 * 2**20, max(os.path.getsize(seq_path), 64 * 2**20))
        rand_file_size = (rand_file_size // _RAND_BLOCK) * _RAND_BLOCK
        with open(rand_path, "wb", buffering=0) as f:
            f.write(b"\x00" * rand_file_size)
        os.sync() if hasattr(os, "sync") else None
        rand_w = _rand_write(rand_path, rand_file_size, _RAND_BLOCK, _DUR)
        emit(
            {
                "type": "progress",
                "test": "write",
                "pct": 78,
                "message": f"Rand write: {rand_w:.0f} IOPS — random read…",
                "rand_write_iops": round(rand_w, 0),
            }
        )
        rand_r = _rand_read(rand_path, rand_file_size, _RAND_BLOCK, _DUR)
        emit(
            {
                "type": "progress",
                "test": "write",
                "pct": 97,
                "message": f"Rand read: {rand_r:.0f} IOPS — computing score…",
                "rand_read_iops": round(rand_r, 0),
            }
        )
    score = log_score(seq_r, _FLOOR, _CEIL)
    result = {
        "type": "result",
        "test": "write",
        "seq_write_mbs": round(seq_w, 1),
        "seq_read_mbs": round(seq_r, 1),
        "rand_write_iops": round(rand_w, 0),
        "rand_read_iops": round(rand_r, 0),
        "rand_block_kb": _RAND_BLOCK // 1024,
        "score": round(score, 1),
        "thermal_throttled": False,
    }
    emit(result)
    return result


if __name__ == "__main__":
    main()
