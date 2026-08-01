"""
DistribAI Benchmark: Memory Efficiency
=======================================
Measures model swapping, checkpointing performance, and memory fragmentation handling.
These metrics determine how efficiently a node can handle large models and memory-intensive tasks.
"""

import json
import math
import os
import tempfile
import time

_FLOOR = float(os.environ.get("SCORE_FLOOR_MBPS", 1.0))
_CEIL = float(os.environ.get("SCORE_CEIL_MBPS", 10000.0))

try:
    import numpy as np
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
    _HAS_NUMPY = True
except ImportError:
    torch = None
    nn = None
    np = None
    _HAS_TORCH = _HAS_NUMPY = False


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def create_test_models():
    """Create test models of different sizes."""
    models = {}

    # Small model
    models["small"] = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

    # Medium model
    models["medium"] = nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)
    )

    # Large model
    models["large"] = nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    return models


def benchmark_model_swapping():
    """Benchmark model swapping performance."""
    if not _HAS_TORCH:
        emit({"type": "skip", "test": "memory_efficiency", "reason": "PyTorch not available"})
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = create_test_models()

    # Move models to device
    for model in models.values():
        model.to(device)

    # Benchmark model swapping
    swap_times = []
    num_swaps = 20

    for i in range(num_swaps):
        # Cycle through models
        model_names = ["small", "medium", "large"]
        current_model = models[model_names[i % 3]]

        # Simulate model loading (move to device if not already)
        start_time = time.time()
        current_model.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        swap_times.append(end_time - start_time)

        progress = int((i + 1) / num_swaps * 33)
        if i % 5 == 0:
            emit(
                {
                    "type": "progress",
                    "test": "memory_efficiency",
                    "pct": progress,
                    "message": f"Model swap {i + 1}/{num_swaps}: {end_time - start_time:.3f}s",
                }
            )

    # Calculate average swap time and throughput
    avg_swap_time = sum(swap_times) / len(swap_times)
    swap_throughput = 1.0 / avg_swap_time  # Swaps per second

    emit(
        {
            "type": "progress",
            "test": "memory_efficiency",
            "pct": 33,
            "message": f"Model swapping: {swap_throughput:.1f} swaps/sec",
        }
    )

    return log_score(swap_throughput, _FLOOR, _CEIL)


def benchmark_checkpointing():
    """Benchmark checkpointing performance."""
    if not _HAS_TORCH:
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_test_models()["large"].to(device)

    # Create optimizer and state
    optimizer = torch.optim.Adam(model.parameters())

    # Create checkpoint data
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 100,
        "loss": 0.123,
        "metadata": {"version": "1.0", "timestamp": time.time()},
    }

    # Benchmark checkpoint saving
    save_times = []
    num_saves = 10

    for i in range(num_saves):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            checkpoint_file = f.name

        try:
            start_time = time.time()
            torch.save(checkpoint, checkpoint_file)
            end_time = time.time()

            save_times.append(end_time - start_time)

            # Test loading
            load_start = time.time()
            torch.load(checkpoint_file)
            load_end = time.time()

            os.unlink(checkpoint_file)

            progress = 33 + int((i + 1) / num_saves * 33)
            if i % 3 == 0:
                emit(
                    {
                        "type": "progress",
                        "test": "memory_efficiency",
                        "pct": progress,
                        "message": f"Checkpoint {i + 1}/{num_saves}: save {end_time - start_time:.3f}s, load {load_end - load_start:.3f}s",
                    }
                )

        except Exception as e:
            emit(
                {
                    "type": "progress",
                    "test": "memory_efficiency",
                    "pct": 66,
                    "message": f"Checkpoint error: {str(e)}",
                }
            )

    # Calculate average checkpointing throughput
    avg_save_time = sum(save_times) / len(save_times)
    checkpoint_throughput = 1.0 / avg_save_time

    emit(
        {
            "type": "progress",
            "test": "memory_efficiency",
            "pct": 66,
            "message": f"Checkpointing: {checkpoint_throughput:.1f} checkpoints/sec",
        }
    )

    return log_score(checkpoint_throughput, _FLOOR, _CEIL)


def benchmark_memory_fragmentation():
    """Benchmark memory fragmentation handling."""
    if not _HAS_TORCH:
        return 0.0

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test memory allocation patterns that cause fragmentation
    fragmentation_scores = []
    num_tests = 10

    for i in range(num_tests):
        # Create tensors of varying sizes (fragmentation pattern)
        tensors = []
        sizes = [1024, 2048, 512, 4096, 256, 8192, 128, 16384]  # Varying sizes

        start_time = time.time()

        # Allocate tensors
        for size in sizes:
            if torch.cuda.is_available():
                tensor = torch.randn(size, size).cuda()
            else:
                tensor = torch.randn(size, size)
            tensors.append(tensor)

        # Deallocate in random order (causes fragmentation)
        import random

        random.shuffle(tensors)

        for tensor in tensors:
            del tensor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        end_time = time.time()

        # Measure allocation/deallocation speed
        fragmentation_score = len(sizes) / (end_time - start_time)
        fragmentation_scores.append(fragmentation_score)

        progress = 66 + int((i + 1) / num_tests * 34)
        if i % 3 == 0:
            emit(
                {
                    "type": "progress",
                    "test": "memory_efficiency",
                    "pct": progress,
                    "message": f"Fragmentation test {i + 1}/{num_tests}: {fragmentation_score:.1f} ops/sec",
                }
            )

    # Calculate average fragmentation handling score
    avg_fragmentation_score = sum(fragmentation_scores) / len(fragmentation_scores)

    emit(
        {
            "type": "progress",
            "test": "memory_efficiency",
            "pct": 100,
            "message": f"Memory fragmentation handling: {avg_fragmentation_score:.1f} ops/sec",
        }
    )

    return log_score(avg_fragmentation_score, _FLOOR, _CEIL)


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "memory_efficiency",
            "display": "Memory Efficiency",
            "message": "Testing model swapping, checkpointing, and memory fragmentation...",
        }
    )

    if not _HAS_TORCH:
        emit({"type": "skip", "test": "memory_efficiency", "reason": "PyTorch not available"})
        return

    # Run sub-benchmarks
    swapping_score = benchmark_model_swapping()
    checkpointing_score = benchmark_checkpointing()
    fragmentation_score = benchmark_memory_fragmentation()

    # Calculate overall score
    scores = [swapping_score, checkpointing_score, fragmentation_score]
    valid_scores = [s for s in scores if s > 0]

    if not valid_scores:
        overall_score = 0.0
    else:
        overall_score = sum(valid_scores) / len(valid_scores)

    emit(
        {
            "type": "result",
            "test": "memory_efficiency",
            "score": overall_score,
            "swapping_score": swapping_score,
            "checkpointing_score": checkpointing_score,
            "fragmentation_score": fragmentation_score,
            "thermal_throttled": False,
        }
    )


if __name__ == "__main__":
    main()
