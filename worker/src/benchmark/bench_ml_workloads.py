"""
DistribAI Benchmark: ML Workloads
===============================
Measures model loading time, inference throughput, and mixed precision performance.
These metrics help determine suitability for different AI training tasks.
"""

import json
import math
import os
import time

_FLOOR = float(os.environ.get("SCORE_FLOOR_MBPS", 1.0))
_CEIL = float(os.environ.get("SCORE_CEIL_MBPS", 1000.0))

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models

    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    nn = None
    models = None
    _HAS_TORCH = _HAS_CUDA = False


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def create_test_model(model_size: str) -> nn.Module:
    """Create a test model of specified size."""
    if model_size == "small":
        return nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10)
        )
    elif model_size == "medium":
        return nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    else:  # large
        return nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )


def benchmark_model_loading():
    """Benchmark model loading time."""
    if not _HAS_TORCH:
        emit({"type": "skip", "test": "ml_workloads", "reason": "PyTorch not available"})
        return 0.0

    device = torch.device("cuda" if _HAS_CUDA else "cpu")
    load_times = {}

    for size in ["small", "medium", "large"]:
        model = create_test_model(size)
        model.to(device)

        # Time the loading process
        start_time = time.time()
        torch.save(model.state_dict(), f"/tmp/test_model_{size}.pth")
        loaded_model = create_test_model(size)
        loaded_model.load_state_dict(torch.load(f"/tmp/test_model_{size}.pth"))
        loaded_model.to(device)
        load_time = time.time() - start_time

        load_times[size] = load_time
        emit(
            {
                "type": "progress",
                "test": "ml_workloads",
                "pct": 20,
                "message": f"{size.capitalize()} model load time: {load_time:.3f}s",
            }
        )

    # Score based on inverse of average load time (faster = better)
    avg_load_time = sum(load_times.values()) / len(load_times)
    throughput_score = log_score(1.0 / avg_load_time, _FLOOR, _CEIL)

    return throughput_score


def benchmark_inference():
    """Benchmark inference throughput."""
    if not _HAS_TORCH:
        return 0.0

    device = torch.device("cuda" if _HAS_CUDA else "cpu")
    model = create_test_model("medium")
    model.to(device)
    model.eval()

    # Create test data
    batch_size = 64
    input_data = torch.randn(batch_size, 784).to(device)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    # Benchmark inference
    if _HAS_CUDA:
        torch.cuda.synchronize()

    start_time = time.time()
    num_runs = 100

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)

    if _HAS_CUDA:
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time

    samples_per_second = (batch_size * num_runs) / total_time
    emit(
        {
            "type": "progress",
            "test": "ml_workloads",
            "pct": 60,
            "message": f"Inference throughput: {samples_per_second:.1f} samples/sec",
        }
    )

    return log_score(samples_per_second, _FLOOR, _CEIL)


def benchmark_mixed_precision():
    """Benchmark mixed precision performance."""
    if not _HAS_TORCH or not _HAS_CUDA:
        emit(
            {
                "type": "skip",
                "test": "ml_workloads",
                "reason": "CUDA not available for mixed precision",
            }
        )
        return 0.0

    device = torch.device("cuda")

    # Test different precisions
    precisions = ["fp32", "fp16", "bf16"] if torch.cuda.is_bf16_supported() else ["fp32", "fp16"]
    throughput_scores = {}

    for precision in precisions:
        if precision == "fp16":
            dtype = torch.float16
            model = create_test_model("medium").half()
        elif precision == "bf16":
            dtype = torch.bfloat16
            model = create_test_model("medium").bfloat16()
        else:  # fp32
            dtype = torch.float32
            model = create_test_model("medium")

        model.to(device)
        model.eval()

        input_data = torch.randn(64, 784).to(device).to(dtype)

        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_data)

        torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for _ in range(50):
                _ = model(input_data)

        torch.cuda.synchronize()
        end_time = time.time()

        throughput = (64 * 50) / (end_time - start_time)
        throughput_scores[precision] = throughput

        emit(
            {
                "type": "progress",
                "test": "ml_workloads",
                "pct": 80,
                "message": f"{precision.upper()} throughput: {throughput:.1f} samples/sec",
            }
        )

    # Score based on best precision performance
    best_throughput = max(throughput_scores.values())
    return log_score(best_throughput, _FLOOR, _CEIL)


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "ml_workloads",
            "display": "ML Workloads",
            "message": "Testing model loading, inference, and mixed precision performance...",
        }
    )

    if not _HAS_TORCH:
        emit({"type": "skip", "test": "ml_workloads", "reason": "PyTorch not available"})
        return

    # Run sub-benchmarks
    loading_score = benchmark_model_loading()
    inference_score = benchmark_inference()
    mixed_precision_score = benchmark_mixed_precision()

    # Calculate overall score
    scores = [loading_score, inference_score, mixed_precision_score]
    valid_scores = [s for s in scores if s > 0]

    if not valid_scores:
        overall_score = 0.0
    else:
        overall_score = sum(valid_scores) / len(valid_scores)

    emit(
        {
            "type": "result",
            "test": "ml_workloads",
            "score": overall_score,
            "loading_score": loading_score,
            "inference_score": inference_score,
            "mixed_precision_score": mixed_precision_score,
            "thermal_throttled": False,
        }
    )


if __name__ == "__main__":
    main()
