"""Unit tests for worker benchmark registry (no GPU execution)."""

from __future__ import annotations


def test_bench_runner_registry_lists_core_benchmarks():
    from worker.src.benchmark import bench_runner

    names = {entry[0] for entry in bench_runner._BENCHMARKS}
    assert "tensor" in names
    assert "vram" in names
    assert "network" in names


def test_bench_runner_weights_sum_to_one():
    from worker.src.benchmark import bench_runner

    total = sum(bench_runner._WEIGHTS.values())
    assert abs(total - 1.0) < 0.01
