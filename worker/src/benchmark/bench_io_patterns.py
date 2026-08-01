"""
DistribAI Benchmark: I/O Patterns
================================
Measures sequential vs random access patterns, small vs large file performance.
These metrics determine storage efficiency for different training data access patterns.
"""

import json
import math
import os
import random
import tempfile
import time

_FLOOR = float(os.environ.get("SCORE_FLOOR_MBPS", 1.0))
_CEIL = float(os.environ.get("SCORE_CEIL_MBPS", 1000.0))

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def create_test_file(size_mb: int, pattern: str = "sequential") -> str:
    """Create a test file with specified access pattern."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filename = f.name

        if pattern == "sequential":
            # Write sequential data
            data = b"A" * 1024 * 1024  # 1MB chunk
            for _ in range(size_mb):
                f.write(data)
        else:  # random pattern
            # Write data with random offsets
            file_size = size_mb * 1024 * 1024
            f.seek(file_size - 1)
            f.write(b"\0")
            f.seek(0)

            # Write random data at random positions
            for _ in range(size_mb // 10):  # 10% of file size
                pos = random.randint(0, file_size - 1024)
                f.seek(pos)
                f.write(os.urandom(1024))

        return filename


def benchmark_sequential_read():
    """Benchmark sequential file reading."""
    if not _HAS_NUMPY:
        emit({"type": "skip", "test": "io_patterns", "reason": "NumPy not available"})
        return 0.0

    file_sizes = [10, 50, 100]  # MB
    read_scores = []

    for i, size_mb in enumerate(file_sizes):
        filename = create_test_file(size_mb, "sequential")

        try:
            # Time sequential read
            start_time = time.time()

            with open(filename, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break

            end_time = time.time()
            read_time = end_time - start_time

            # Calculate read throughput (MB/sec)
            throughput = size_mb / read_time
            read_scores.append(throughput)

            progress = (i + 1) * 20
            emit(
                {
                    "type": "progress",
                    "test": "io_patterns",
                    "pct": progress,
                    "message": f"Sequential read ({size_mb}MB): {throughput:.1f} MB/sec",
                }
            )

        finally:
            os.unlink(filename)

    avg_throughput = sum(read_scores) / len(read_scores)
    return log_score(avg_throughput, _FLOOR, _CEIL)


def benchmark_random_read():
    """Benchmark random file reading."""
    if not _HAS_NUMPY:
        return 0.0

    file_size_mb = 100
    filename = create_test_file(file_size_mb, "random")

    try:
        # Time random reads
        start_time = time.time()
        num_reads = 1000
        read_size = 4096  # 4KB reads

        with open(filename, "rb") as f:
            file_size = os.path.getsize(filename)

            for _ in range(num_reads):
                # Random position
                pos = random.randint(0, file_size - read_size)
                f.seek(pos)
                f.read(read_size)

        end_time = time.time()
        read_time = end_time - start_time

        # Calculate random read throughput (MB/sec)
        total_data_mb = (num_reads * read_size) / (1024 * 1024)
        throughput = total_data_mb / read_time

        emit(
            {
                "type": "progress",
                "test": "io_patterns",
                "pct": 60,
                "message": f"Random read: {throughput:.1f} MB/sec",
            }
        )

        return log_score(throughput, _FLOOR, _CEIL)

    finally:
        os.unlink(filename)


def benchmark_small_file_ops():
    """Benchmark small file operations."""
    if not _HAS_NUMPY:
        return 0.0

    num_files = 1000
    file_size_kb = 4  # 4KB files

    # Create small files
    filenames = []
    for _i in range(num_files):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(os.urandom(file_size_kb * 1024))
            filenames.append(f.name)

    try:
        # Time small file reads
        start_time = time.time()

        for filename in filenames:
            with open(filename, "rb") as f:
                f.read()

        end_time = time.time()
        read_time = end_time - start_time

        # Calculate small file throughput (files/sec)
        throughput = num_files / read_time

        emit(
            {
                "type": "progress",
                "test": "io_patterns",
                "pct": 80,
                "message": f"Small files: {throughput:.1f} files/sec",
            }
        )

        return log_score(throughput, _FLOOR, _CEIL)

    finally:
        for filename in filenames:
            os.unlink(filename)


def benchmark_large_file_ops():
    """Benchmark large file operations."""
    if not _HAS_NUMPY:
        return 0.0

    file_size_mb = 500  # 500MB file
    filename = create_test_file(file_size_mb, "sequential")

    try:
        # Time large file operations
        start_time = time.time()

        # Read in large chunks
        with open(filename, "rb") as f:
            while True:
                chunk = f.read(16 * 1024 * 1024)  # 16MB chunks
                if not chunk:
                    break

        end_time = time.time()
        read_time = end_time - start_time

        # Calculate large file throughput (MB/sec)
        throughput = file_size_mb / read_time

        emit(
            {
                "type": "progress",
                "test": "io_patterns",
                "pct": 95,
                "message": f"Large file: {throughput:.1f} MB/sec",
            }
        )

        return log_score(throughput, _FLOOR, _CEIL)

    finally:
        os.unlink(filename)


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "io_patterns",
            "display": "I/O Patterns",
            "message": "Testing sequential vs random access, small vs large file performance...",
        }
    )

    if not _HAS_NUMPY:
        emit({"type": "skip", "test": "io_patterns", "reason": "NumPy not available"})
        return

    # Run sub-benchmarks
    sequential_score = benchmark_sequential_read()
    random_score = benchmark_random_read()
    small_file_score = benchmark_small_file_ops()
    large_file_score = benchmark_large_file_ops()

    # Calculate overall score
    scores = [sequential_score, random_score, small_file_score, large_file_score]
    valid_scores = [s for s in scores if s > 0]

    if not valid_scores:
        overall_score = 0.0
    else:
        overall_score = sum(valid_scores) / len(valid_scores)

    emit(
        {
            "type": "result",
            "test": "io_patterns",
            "score": overall_score,
            "sequential_score": sequential_score,
            "random_score": random_score,
            "small_file_score": small_file_score,
            "large_file_score": large_file_score,
            "thermal_throttled": False,
        }
    )


if __name__ == "__main__":
    main()
