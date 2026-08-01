"""Import and unit-test benchmark helpers without running long GPU/network benches."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parents[2] / "worker" / "src" / "benchmark"
_BENCH_MODULES = sorted(
    p.stem for p in _BENCH_DIR.glob("bench_*.py") if p.name != "bench_runner.py"
)


@pytest.mark.parametrize("module_name", _BENCH_MODULES)
def test_benchmark_module_imports(module_name: str):
    mod = importlib.import_module(f"worker.src.benchmark.{module_name}")
    assert hasattr(mod, "emit")
    assert hasattr(mod, "log_score")


@pytest.mark.parametrize(
    "value,floor,ceil,min_score",
    [
        (0.0, 1.0, 10.0, 0.0),
        (10.0, 1.0, 10.0, 99.0),
        (-1.0, 1.0, 10.0, 0.0),
    ],
)
def test_log_score_monotonic(value: float, floor: float, ceil: float, min_score: float):
    from worker.src.benchmark import bench_network

    score = bench_network.log_score(value, floor, ceil)
    assert 0.0 <= score <= 100.0
    if min_score == 0.0:
        assert score == 0.0
    else:
        assert score >= min_score


def test_bench_network_invalid_url_rejected():
    from worker.src.benchmark.bench_network import _try_download

    with pytest.raises(ValueError, match="Invalid benchmark URL"):
        _try_download("http://not-https.example/", 0.1)


def test_bench_runner_compute_overall_renormalizes_weights():
    from worker.src.benchmark import bench_runner

    results = {
        "tensor": {"score": 80.0},
        "vram": {"score": 60.0},
    }
    overall = bench_runner.compute_overall(results)
    assert 0.0 < overall <= 100.0


def test_bench_runner_assign_tier_cpu_only():
    from worker.src.benchmark import bench_runner

    tier = bench_runner.assign_tier(
        25.0,
        {
            "tensor": {"score": 10.0, "gpu_used": False},
            "vram": {"score": 5.0, "max_usable_gb": 0.5},
            "network": {"score": 20.0},
        },
    )
    assert "tier" in tier
    assert tier["tier"]


def test_bench_runner_estimate_throughput_and_tips():
    from worker.src.benchmark import bench_runner

    assert "samples/sec" in bench_runner.estimate_throughput(50.0, "samples")
    tips = bench_runner.get_optimization_tips({"memory": {"score": 10.0}})
    assert any("RAM" in tip or "memory" in tip.lower() for tip in tips)
    avoid = bench_runner.get_avoid_workloads(
        {"tier": "CPU Processing Node", "constraints": {"requires_gpu": False, "max_model_size_gb": 2}}
    )
    assert avoid


def test_bench_data_pipeline_generate_sample_data(tmp_path):
    from worker.src.benchmark.bench_data_pipeline import generate_sample_data

    path = generate_sample_data(1)
    try:
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0
    finally:
        Path(path).unlink(missing_ok=True)


def test_bench_io_patterns_create_test_file(tmp_path):
    from worker.src.benchmark.bench_io_patterns import create_test_file

    path = create_test_file(1, pattern="sequential")
    try:
        assert Path(path).exists()
    finally:
        Path(path).unlink(missing_ok=True)


def test_bench_memory_log_score_handles_none():
    from worker.src.benchmark.bench_memory import log_score

    assert log_score(None, 1.0, 10.0) == 0.0


def test_bench_runner_emit_json(capsys):
    from worker.src.benchmark.bench_runner import emit

    emit({"type": "ping", "ok": True})
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["type"] == "ping"
