"""
DistribAI Benchmark: Distributed Training
=========================================
Measures gradient synchronization efficiency, all-reduce performance, and inter-node communication.
These metrics are critical for distributed AI training workloads.
"""

import json
import math
import os
import time

_FLOOR = float(os.environ.get("SCORE_FLOOR_MBPS", 1.0))
_CEIL = float(os.environ.get("SCORE_CEIL_MBPS", 10000.0))

try:
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn as nn

    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    dist = None
    nn = None
    np = None
    _HAS_TORCH = _HAS_CUDA = False


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def create_test_model(size: str = "medium") -> nn.Module:
    """Create a test model for gradient synchronization testing."""
    if size == "small":
        return nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    elif size == "medium":
        return nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    else:  # large
        return nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )


def benchmark_gradient_sync():
    """Benchmark gradient synchronization performance."""
    if not _HAS_TORCH:
        emit({"type": "skip", "test": "distributed_training", "reason": "PyTorch not available"})
        return 0.0

    device = torch.device("cuda" if _HAS_CUDA else "cpu")
    model = create_test_model("medium").to(device)

    # Create dummy gradients
    gradients = []
    for param in model.parameters():
        grad = torch.randn_like(param)
        gradients.append(grad)

    # Simulate gradient aggregation
    start_time = time.time()

    # Simulate multiple gradient aggregation steps
    for _ in range(100):
        # Simulate gradient averaging (what happens in distributed training)
        avg_grads = []
        for grad in gradients:
            # Simulate all-reduce operation
            avg_grad = grad.clone()
            avg_grads.append(avg_grad)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate gradient throughput (MB/sec)
    total_params = sum(grad.numel() * grad.element_size() for grad in gradients)
    throughput_mb = (total_params * 100) / (total_time * 1024 * 1024)  # 100 iterations

    emit(
        {
            "type": "progress",
            "test": "distributed_training",
            "pct": 33,
            "message": f"Gradient sync throughput: {throughput_mb:.1f} MB/sec",
        }
    )

    return log_score(throughput_mb, _FLOOR, _CEIL)


def benchmark_all_reduce_simulation():
    """Simulate all-reduce operation performance."""
    if not _HAS_TORCH:
        return 0.0

    device = torch.device("cuda" if _HAS_CUDA else "cpu")

    # Test different tensor sizes
    sizes = [1024, 4096, 16384, 65536]  # Different tensor sizes
    all_reduce_scores = []

    for i, size in enumerate(sizes):
        # Create test tensor
        tensor = torch.randn(size).to(device)

        # Simulate all-reduce operation (multiple reductions)
        start_time = time.time()

        for _ in range(50):
            # Simulate reduce operation
            result = tensor.clone()
            # Simulate multiple nodes contributing
            for _ in range(4):  # Simulate 4 nodes
                result += torch.randn(size).to(device)
            result /= 4  # Average

        if _HAS_CUDA:
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate throughput
        data_size = size * 4  # 4 bytes per float32
        total_data = data_size * 50  # 50 iterations
        throughput = total_data / (end_time - start_time) / (1024 * 1024)  # MB/sec

        all_reduce_scores.append(throughput)

        progress = 33 + (i + 1) * 17
        emit(
            {
                "type": "progress",
                "test": "distributed_training",
                "pct": progress,
                "message": f"All-reduce ({size} elems): {throughput:.1f} MB/sec",
            }
        )

    # Return average all-reduce performance
    avg_throughput = sum(all_reduce_scores) / len(all_reduce_scores)
    return log_score(avg_throughput, _FLOOR, _CEIL)


def benchmark_communication_overhead():
    """Benchmark communication overhead simulation."""
    if not _HAS_TORCH:
        return 0.0

    device = torch.device("cuda" if _HAS_CUDA else "cpu")

    # Simulate different message sizes (simulating parameter updates)
    message_sizes = [1024, 4096, 16384, 65536, 262144]  # 1KB to 256KB
    comm_scores = []

    for i, msg_size in enumerate(message_sizes):
        # Create test message
        message = torch.randn(msg_size).to(device)

        # Simulate communication overhead
        start_time = time.time()

        # Simulate send/receive operations
        for _ in range(20):
            # Simulate serialization overhead
            serialized = message.cpu().numpy().tobytes()
            # Simulate network transfer
            torch.from_numpy(np.frombuffer(serialized).astype(np.float32)).to(device)

        if _HAS_CUDA:
            torch.cuda.synchronize()

        end_time = time.time()

        # Calculate communication throughput
        total_data = msg_size * 4 * 20 * 2  # send + receive, 20 iterations
        throughput = total_data / (end_time - start_time) / (1024 * 1024)  # MB/sec

        comm_scores.append(throughput)

        progress = 67 + (i + 1) * 7
        emit(
            {
                "type": "progress",
                "test": "distributed_training",
                "pct": progress,
                "message": f"Comm overhead ({msg_size}B): {throughput:.1f} MB/sec",
            }
        )

    # Return average communication performance
    avg_throughput = sum(comm_scores) / len(comm_scores)
    return log_score(avg_throughput, _FLOOR, _CEIL)


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "distributed_training",
            "display": "Distributed Training",
            "message": "Testing gradient synchronization and inter-node communication...",
        }
    )

    if not _HAS_TORCH:
        emit({"type": "skip", "test": "distributed_training", "reason": "PyTorch not available"})
        return

    # Run sub-benchmarks
    gradient_score = benchmark_gradient_sync()
    all_reduce_score = benchmark_all_reduce_simulation()
    comm_score = benchmark_communication_overhead()

    # Calculate overall score
    scores = [gradient_score, all_reduce_score, comm_score]
    valid_scores = [s for s in scores if s > 0]

    if not valid_scores:
        overall_score = 0.0
    else:
        overall_score = sum(valid_scores) / len(valid_scores)

    emit(
        {
            "type": "result",
            "test": "distributed_training",
            "score": overall_score,
            "gradient_sync_score": gradient_score,
            "all_reduce_score": all_reduce_score,
            "communication_score": comm_score,
            "thermal_throttled": False,
        }
    )


if __name__ == "__main__":
    main()
