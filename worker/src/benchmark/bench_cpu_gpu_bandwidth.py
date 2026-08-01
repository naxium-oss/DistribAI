"""
DistribAI Benchmark: CPU-GPU Bandwidth
=====================================
Measures PCIe transfer speeds and host-device synchronization performance.
These metrics determine how efficiently data moves between CPU and GPU memory.
"""

import json
import math
import os
import time

_FLOOR = float(os.environ.get("SCORE_FLOOR_MBPS", 1.0))
_CEIL = float(os.environ.get("SCORE_CEIL_MBPS", 50000.0))  # 50GB/s for modern PCIe

try:
    import numpy as np
    import torch

    _HAS_TORCH = True
    _HAS_NUMPY = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    np = None
    _HAS_TORCH = _HAS_NUMPY = _HAS_CUDA = False


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def benchmark_cpu_to_gpu_transfer():
    """Benchmark CPU to GPU data transfer."""
    if not _HAS_TORCH or not _HAS_CUDA:
        emit({"type": "skip", "test": "cpu_gpu_bandwidth", "reason": "CUDA not available"})
        return 0.0

    # Test different tensor sizes
    tensor_sizes = [1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024]  # 1MB, 10MB, 100MB
    transfer_scores = []

    for i, size in enumerate(tensor_sizes):
        # Create CPU tensor
        cpu_tensor = torch.randn(size)

        # Time transfer to GPU
        start_time = time.time()
        cpu_tensor.cuda()
        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate transfer throughput (MB/sec)
        data_size_mb = size * 4 / (1024 * 1024)  # 4 bytes per float32
        transfer_time = end_time - start_time
        throughput = data_size_mb / transfer_time

        transfer_scores.append(throughput)

        progress = (i + 1) * 20
        emit(
            {
                "type": "progress",
                "test": "cpu_gpu_bandwidth",
                "pct": progress,
                "message": f"CPU→GPU ({data_size_mb:.1f}MB): {throughput:.1f} MB/sec",
            }
        )

    avg_throughput = sum(transfer_scores) / len(transfer_scores)
    return log_score(avg_throughput, _FLOOR, _CEIL)


def benchmark_gpu_to_cpu_transfer():
    """Benchmark GPU to CPU data transfer."""
    if not _HAS_TORCH or not _HAS_CUDA:
        return 0.0

    # Test different tensor sizes
    tensor_sizes = [1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024]  # 1MB, 10MB, 100MB
    transfer_scores = []

    for i, size in enumerate(tensor_sizes):
        # Create GPU tensor
        gpu_tensor = torch.randn(size).cuda()

        # Time transfer to CPU
        start_time = time.time()
        gpu_tensor.cpu()
        torch.cuda.synchronize()
        end_time = time.time()

        # Calculate transfer throughput (MB/sec)
        data_size_mb = size * 4 / (1024 * 1024)  # 4 bytes per float32
        transfer_time = end_time - start_time
        throughput = data_size_mb / transfer_time

        transfer_scores.append(throughput)

        progress = 40 + (i + 1) * 20
        emit(
            {
                "type": "progress",
                "test": "cpu_gpu_bandwidth",
                "pct": progress,
                "message": f"GPU→CPU ({data_size_mb:.1f}MB): {throughput:.1f} MB/sec",
            }
        )

    avg_throughput = sum(transfer_scores) / len(transfer_scores)
    return log_score(avg_throughput, _FLOOR, _CEIL)


def benchmark_bidirectional_transfer():
    """Benchmark bidirectional data transfer."""
    if not _HAS_TORCH or not _HAS_CUDA:
        return 0.0

    # Test bidirectional transfer
    tensor_size = 50 * 1024 * 1024  # 50MB
    num_iterations = 10

    cpu_tensor = torch.randn(tensor_size)
    gpu_tensor = torch.randn(tensor_size).cuda()

    # Time bidirectional transfers
    start_time = time.time()

    for _ in range(num_iterations):
        # CPU to GPU
        cpu_tensor.cuda()
        # GPU to CPU
        gpu_tensor.cpu()

        # Synchronize to ensure transfers complete
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate bidirectional throughput
    total_data_mb = (tensor_size * 4 * 2 * num_iterations) / (1024 * 1024)  # 2x for bidirectional
    transfer_time = end_time - start_time
    throughput = total_data_mb / transfer_time

    emit(
        {
            "type": "progress",
            "test": "cpu_gpu_bandwidth",
            "pct": 90,
            "message": f"Bidirectional: {throughput:.1f} MB/sec",
        }
    )

    return log_score(throughput, _FLOOR, _CEIL)


def benchmark_host_device_sync():
    """Benchmark host-device synchronization overhead."""
    if not _HAS_TORCH or not _HAS_CUDA:
        return 0.0

    # Test synchronization overhead
    gpu_tensor = torch.randn(1024 * 1024).cuda()
    num_syncs = 1000

    # Time synchronization operations
    start_time = time.time()

    for _ in range(num_syncs):
        # Perform GPU operation
        torch.sum(gpu_tensor)
        # Synchronize
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate sync throughput (syncs/sec)
    sync_time = end_time - start_time
    sync_throughput = num_syncs / sync_time

    emit(
        {
            "type": "progress",
            "test": "cpu_gpu_bandwidth",
            "pct": 95,
            "message": f"Sync overhead: {sync_throughput:.0f} syncs/sec",
        }
    )

    # Higher sync throughput is better (lower overhead)
    return log_score(sync_throughput, _FLOOR, _CEIL)


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "cpu_gpu_bandwidth",
            "display": "CPU-GPU Bandwidth",
            "message": "Testing PCIe transfer speeds and host-device synchronization...",
        }
    )

    if not _HAS_TORCH or not _HAS_CUDA:
        emit({"type": "skip", "test": "cpu_gpu_bandwidth", "reason": "CUDA not available"})
        return

    # Run sub-benchmarks
    cpu_to_gpu_score = benchmark_cpu_to_gpu_transfer()
    gpu_to_cpu_score = benchmark_gpu_to_cpu_transfer()
    bidirectional_score = benchmark_bidirectional_transfer()
    sync_score = benchmark_host_device_sync()

    # Calculate overall score
    scores = [cpu_to_gpu_score, gpu_to_cpu_score, bidirectional_score, sync_score]
    valid_scores = [s for s in scores if s > 0]

    if not valid_scores:
        overall_score = 0.0
    else:
        overall_score = sum(valid_scores) / len(valid_scores)

    emit(
        {
            "type": "result",
            "test": "cpu_gpu_bandwidth",
            "score": overall_score,
            "cpu_to_gpu_score": cpu_to_gpu_score,
            "gpu_to_cpu_score": gpu_to_cpu_score,
            "bidirectional_score": bidirectional_score,
            "sync_score": sync_score,
            "thermal_throttled": False,
        }
    )


if __name__ == "__main__":
    main()
