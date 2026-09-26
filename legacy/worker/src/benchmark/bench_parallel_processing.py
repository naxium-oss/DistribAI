"""
DistribAI Benchmark: Parallel Processing
=======================================
Measures multi-GPU scaling, threading performance, and batch sizing efficiency.
These metrics determine how well a node can handle parallel training workloads.
"""

import json
import math
import os
import time

_FLOOR = float(os.environ.get("SCORE_FLOOR_MBPS", 1.0))
_CEIL = float(os.environ.get("SCORE_CEIL_MBPS", 1000.0))

try:
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
    _HAS_NUMPY = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    nn = None
    np = None
    ThreadPoolExecutor = None
    _HAS_TORCH = _HAS_NUMPY = _HAS_CUDA = False


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def create_test_model():
    """Create a test model for parallel processing."""
    return nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)
    )


def benchmark_multi_gpu_scaling():
    """Benchmark multi-GPU scaling if available."""
    if not _HAS_TORCH or not _HAS_CUDA:
        emit({"type": "skip", "test": "parallel_processing", "reason": "CUDA not available"})
        return 0.0

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        emit(
            {
                "type": "progress",
                "test": "parallel_processing",
                "pct": 25,
                "message": "Single GPU detected - testing single GPU performance",
            }
        )
        return 50.0  # Default score for single GPU

    # Test multi-GPU scaling
    model = create_test_model()
    batch_size = 64

    # Single GPU baseline
    device0 = torch.device("cuda:0")
    model0 = model.to(device0)
    input_data = torch.randn(batch_size, 784).to(device0)

    start_time = time.time()
    for _ in range(100):
        model0(input_data)
        torch.cuda.synchronize()
    single_gpu_time = time.time() - start_time

    # Multi-GPU test
    devices = [torch.device(f"cuda:{i}") for i in range(min(num_gpus, 2))]  # Test up to 2 GPUs
    models = [model.to(device) for device in devices]
    inputs = [torch.randn(batch_size, 784).to(device) for device in devices]

    start_time = time.time()
    for model, input_data in zip(models, inputs, strict=False):
        for _ in range(50):  # Half the iterations per GPU
            model(input_data)
            torch.cuda.synchronize()
    multi_gpu_time = time.time() - start_time

    # Calculate scaling efficiency
    scaling_factor = single_gpu_time / multi_gpu_time
    ideal_scaling = len(devices)
    efficiency = (scaling_factor / ideal_scaling) * 100

    emit(
        {
            "type": "progress",
            "test": "parallel_processing",
            "pct": 25,
            "message": f"Multi-GPU scaling: {scaling_factor:.2f}x, Efficiency: {efficiency:.1f}%",
        }
    )

    return log_score(efficiency, _FLOOR, _CEIL)


def benchmark_threading_performance():
    """Benchmark threading performance."""
    if not _HAS_NUMPY or not ThreadPoolExecutor:
        emit({"type": "skip", "test": "parallel_processing", "reason": "Threading not available"})
        return 0.0

    # Test different thread counts
    thread_counts = [1, 2, 4, 8]
    threading_scores = []

    def compute_task(size):
        """Simple compute task for threading test."""
        data = np.random.randn(size)
        result = np.fft.fft(data)
        return np.abs(result)

    task_size = 100000
    num_tasks = 20

    for i, num_threads in enumerate(thread_counts):
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(compute_task, task_size) for _ in range(num_tasks)]
            [future.result() for future in futures]

        end_time = time.time()

        # Calculate throughput (tasks/sec)
        throughput = num_tasks / (end_time - start_time)
        threading_scores.append(throughput)

        progress = 25 + (i + 1) * 15
        emit(
            {
                "type": "progress",
                "test": "parallel_processing",
                "pct": progress,
                "message": f"Threading ({num_threads} threads): {throughput:.1f} tasks/sec",
            }
        )

    # Calculate threading efficiency
    baseline_score = threading_scores[0]  # Single thread
    best_score = max(threading_scores)
    threading_efficiency = (best_score / baseline_score) * 100

    emit(
        {
            "type": "progress",
            "test": "parallel_processing",
            "pct": 85,
            "message": f"Threading efficiency: {threading_efficiency:.1f}%",
        }
    )

    return log_score(threading_efficiency, _FLOOR, _CEIL)


def benchmark_batch_scaling():
    """Benchmark batch size scaling efficiency."""
    if not _HAS_TORCH:
        return 0.0

    device = torch.device("cuda" if _HAS_CUDA else "cpu")
    model = create_test_model().to(device)
    model.eval()

    # Test different batch sizes
    batch_sizes = [16, 32, 64, 128, 256]
    batch_scores = []

    for i, batch_size in enumerate(batch_sizes):
        input_data = torch.randn(batch_size, 784).to(device)

        # Warm up
        with torch.no_grad():
            _ = model(input_data)

        if _HAS_CUDA:
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.time()
        num_iterations = max(10, 1000 // batch_size)  # Adjust iterations based on batch size

        with torch.no_grad():
            for _ in range(num_iterations):
                model(input_data)

        if _HAS_CUDA:
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate throughput (samples/sec)
        throughput = (batch_size * num_iterations) / (end_time - start_time)
        batch_scores.append(throughput)

        progress = 85 + (i + 1) * 3
        emit(
            {
                "type": "progress",
                "test": "parallel_processing",
                "pct": progress,
                "message": f"Batch size {batch_size}: {throughput:.1f} samples/sec",
            }
        )

    # Calculate batch scaling efficiency
    best_throughput = max(batch_scores)
    baseline_throughput = batch_scores[0]  # Smallest batch size
    scaling_efficiency = (best_throughput / baseline_throughput) * 100

    emit(
        {
            "type": "progress",
            "test": "parallel_processing",
            "pct": 100,
            "message": f"Batch scaling efficiency: {scaling_efficiency:.1f}%",
        }
    )

    return log_score(scaling_efficiency, _FLOOR, _CEIL)


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "parallel_processing",
            "display": "Parallel Processing",
            "message": "Testing multi-GPU scaling, threading, and batch size efficiency...",
        }
    )

    if not _HAS_TORCH:
        emit({"type": "skip", "test": "parallel_processing", "reason": "PyTorch not available"})
        return

    # Run sub-benchmarks
    multi_gpu_score = benchmark_multi_gpu_scaling()
    threading_score = benchmark_threading_performance()
    batch_score = benchmark_batch_scaling()

    # Calculate overall score
    scores = [multi_gpu_score, threading_score, batch_score]
    valid_scores = [s for s in scores if s > 0]

    if not valid_scores:
        overall_score = 0.0
    else:
        overall_score = sum(valid_scores) / len(valid_scores)

    emit(
        {
            "type": "result",
            "test": "parallel_processing",
            "score": overall_score,
            "multi_gpu_score": multi_gpu_score,
            "threading_score": threading_score,
            "batch_score": batch_score,
            "thermal_throttled": False,
        }
    )


if __name__ == "__main__":
    main()
