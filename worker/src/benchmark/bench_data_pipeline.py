"""
DistribAI Benchmark: Data Pipeline
=================================
Measures dataset loading speed, tokenization throughput, and preprocessing performance.
These metrics determine how efficiently a node can handle data-intensive training tasks.
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
    import torch

    _HAS_TORCH = True
    _HAS_NUMPY = True
except ImportError:
    torch = None
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


def generate_sample_data(size_mb: int) -> str:
    """Generate sample data for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        # Generate realistic text data
        words = [
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "neural",
            "network",
            "deep",
            "model",
            "training",
            "inference",
            "data",
            "processing",
        ]

        target_size = size_mb * 1024 * 1024
        current_size = 0

        while current_size < target_size:
            sentence = " ".join(random.choices(words, k=20)) + ".\n"
            f.write(sentence)
            current_size += len(sentence.encode("utf-8"))

        return f.name


def benchmark_dataset_loading():
    """Benchmark dataset loading performance."""
    if not _HAS_NUMPY:
        emit({"type": "skip", "test": "data_pipeline", "reason": "NumPy not available"})
        return 0.0

    # Create test datasets of different sizes
    dataset_sizes = [1, 5, 10]  # MB
    loading_scores = []

    for i, size_mb in enumerate(dataset_sizes):
        dataset_file = generate_sample_data(size_mb)

        try:
            # Time the loading process
            start_time = time.time()

            # Simulate dataset loading
            loaded_data = []
            with open(dataset_file) as f:
                for line in f:
                    # Simulate processing each line
                    words = line.strip().split()
                    loaded_data.append(words)

            end_time = time.time()
            load_time = end_time - start_time

            # Calculate loading throughput (MB/sec)
            throughput = size_mb / load_time
            loading_scores.append(throughput)

            progress = (i + 1) * 20
            emit(
                {
                    "type": "progress",
                    "test": "data_pipeline",
                    "pct": progress,
                    "message": f"Dataset loading ({size_mb}MB): {throughput:.1f} MB/sec",
                }
            )

        finally:
            # Clean up
            os.unlink(dataset_file)

    # Return average loading performance
    avg_throughput = sum(loading_scores) / len(loading_scores)
    return log_score(avg_throughput, _FLOOR, _CEIL)


def benchmark_tokenization():
    """Benchmark tokenization throughput."""
    if not _HAS_NUMPY:
        return 0.0

    # Generate sample text data
    sample_text = " ".join(["word"] * 10000)  # 10k words
    num_iterations = 100

    # Simple tokenization function
    def simple_tokenize(text):
        return text.split()

    # Benchmark tokenization
    start_time = time.time()

    for _ in range(num_iterations):
        tokens = simple_tokenize(sample_text)

    end_time = time.time()

    # Calculate tokenization throughput (tokens/sec)
    total_tokens = len(tokens) * num_iterations
    throughput = total_tokens / (end_time - start_time)

    emit(
        {
            "type": "progress",
            "test": "data_pipeline",
            "pct": 70,
            "message": f"Tokenization throughput: {throughput:.0f} tokens/sec",
        }
    )

    return log_score(throughput, _FLOOR, _CEIL)


def benchmark_preprocessing():
    """Benchmark data preprocessing performance."""
    if not _HAS_NUMPY:
        return 0.0

    # Generate sample numerical data
    data_size = 100000
    sample_data = np.random.randn(data_size, 10)  # 100k samples, 10 features

    # Common preprocessing operations
    def preprocess_data(data):
        # Normalization
        normalized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

        # Feature scaling
        scaled = normalized / np.max(np.abs(normalized), axis=0)

        # Data augmentation (noise addition)
        augmented = scaled + np.random.normal(0, 0.01, scaled.shape)

        return augmented

    # Benchmark preprocessing
    start_time = time.time()

    for _ in range(10):  # 10 iterations
        preprocess_data(sample_data)

    end_time = time.time()

    # Calculate preprocessing throughput (samples/sec)
    total_samples = data_size * 10
    throughput = total_samples / (end_time - start_time)

    emit(
        {
            "type": "progress",
            "test": "data_pipeline",
            "pct": 85,
            "message": f"Preprocessing throughput: {throughput:.0f} samples/sec",
        }
    )

    return log_score(throughput, _FLOOR, _CEIL)


def benchmark_batch_processing():
    """Benchmark batch processing performance."""
    if not _HAS_TORCH:
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create sample data
    batch_size = 64
    num_batches = 100
    feature_size = 512

    # Generate batch data
    batches = []
    for _ in range(num_batches):
        batch = torch.randn(batch_size, feature_size).to(device)
        batches.append(batch)

    # Benchmark batch processing
    start_time = time.time()

    for batch in batches:
        # Simulate common batch operations
        normalized = (batch - batch.mean()) / batch.std()
        transformed = torch.relu(normalized)
        torch.mean(transformed, dim=0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate batch processing throughput (samples/sec)
    total_samples = batch_size * num_batches
    throughput = total_samples / (end_time - start_time)

    emit(
        {
            "type": "progress",
            "test": "data_pipeline",
            "pct": 95,
            "message": f"Batch processing throughput: {throughput:.0f} samples/sec",
        }
    )

    return log_score(throughput, _FLOOR, _CEIL)


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "data_pipeline",
            "display": "Data Pipeline",
            "message": "Testing dataset loading, tokenization, and preprocessing performance...",
        }
    )

    if not _HAS_NUMPY:
        emit({"type": "skip", "test": "data_pipeline", "reason": "NumPy not available"})
        return

    # Run sub-benchmarks
    loading_score = benchmark_dataset_loading()
    tokenization_score = benchmark_tokenization()
    preprocessing_score = benchmark_preprocessing()
    batch_score = benchmark_batch_processing()

    # Calculate overall score
    scores = [loading_score, tokenization_score, preprocessing_score, batch_score]
    valid_scores = [s for s in scores if s > 0]

    if not valid_scores:
        overall_score = 0.0
    else:
        overall_score = sum(valid_scores) / len(valid_scores)

    emit(
        {
            "type": "result",
            "test": "data_pipeline",
            "score": overall_score,
            "loading_score": loading_score,
            "tokenization_score": tokenization_score,
            "preprocessing_score": preprocessing_score,
            "batch_score": batch_score,
            "thermal_throttled": False,
        }
    )


if __name__ == "__main__":
    main()
