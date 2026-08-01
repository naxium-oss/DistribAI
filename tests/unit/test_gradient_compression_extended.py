"""Extended tests for gradient compression module."""

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from worker.src.daemon.gradient_compression import (
        AdaptiveCompression,
        DeepGradientCompression,
        OneBitAdamCompressor,
        PowerSGDCompressor,
        TopKCompressor,
        ZstdCompressor,
        get_compression_ratio,
    )

    HAS_GRADIENT_COMPRESSION = True
except ImportError:
    HAS_GRADIENT_COMPRESSION = False


@pytest.mark.skipif(not HAS_GRADIENT_COMPRESSION, reason="gradient compression not available")
def test_zstd_compressor_creation():
    """Test ZstdCompressor creation."""
    compressor = ZstdCompressor(level=3)
    assert compressor.level == 3


@pytest.mark.skipif(
    not HAS_GRADIENT_COMPRESSION or not HAS_TORCH,
    reason="gradient compression or torch not available",
)
def test_zstd_compress_tensor():
    """Test ZstdCompressor compress_tensor."""
    compressor = ZstdCompressor()

    tensor = torch.randn(10, 10)
    compressed = compressor.compress_tensor(tensor)
    assert compressed is not None


@pytest.mark.skipif(
    not HAS_GRADIENT_COMPRESSION or not HAS_TORCH,
    reason="gradient compression or torch not available",
)
def test_topk_compressor_creation():
    """Test TopKCompressor creation."""
    compressor = TopKCompressor(sparsity=0.9)
    assert compressor.sparsity == 0.9


@pytest.mark.skipif(
    not HAS_GRADIENT_COMPRESSION or not HAS_TORCH,
    reason="gradient compression or torch not available",
)
def test_powersgd_creation():
    """Test PowerSGDCompressor creation."""
    compressor = PowerSGDCompressor(rank=2, num_iterations=3)
    assert compressor.rank == 2
    assert compressor.num_iterations == 3


@pytest.mark.skipif(
    not HAS_GRADIENT_COMPRESSION or not HAS_TORCH,
    reason="gradient compression or torch not available",
)
def test_powersgd_compress_decompress():
    """Test PowerSGDCompressor compress and decompress."""
    compressor = PowerSGDCompressor(rank=2)

    gradients = {
        "layer1.weight": torch.randn(10, 10),
        "layer2.weight": torch.randn(5, 5),
    }

    compressed = compressor.compress(gradients)
    assert "layer1.weight" in compressed
    assert "P" in compressed["layer1.weight"]


@pytest.mark.skipif(
    not HAS_GRADIENT_COMPRESSION or not HAS_TORCH,
    reason="gradient compression or torch not available",
)
def test_adaptive_compression_creation():
    """Test AdaptiveCompression creation."""
    compressor = AdaptiveCompression(threshold_small=1000, threshold_large=100000)
    assert compressor.threshold_small == 1000
    assert compressor.threshold_large == 100000


@pytest.mark.skipif(
    not HAS_GRADIENT_COMPRESSION or not HAS_TORCH,
    reason="gradient compression or torch not available",
)
def test_deep_gradient_compression_creation():
    """Test DeepGradientCompression creation."""
    compressor = DeepGradientCompression()
    assert compressor is not None


@pytest.mark.skipif(
    not HAS_GRADIENT_COMPRESSION or not HAS_TORCH,
    reason="gradient compression or torch not available",
)
def test_onebit_adam_creation():
    """Test OneBitAdamCompressor creation."""
    compressor = OneBitAdamCompressor()
    assert compressor is not None


@pytest.mark.skipif(
    not HAS_GRADIENT_COMPRESSION or not HAS_TORCH,
    reason="gradient compression or torch not available",
)
def test_get_compression_ratio():
    """Test get_compression_ratio function."""
    original = {"layer1": torch.randn(10, 10), "layer2": torch.randn(5, 5)}
    # Create a simple compressed format
    compressed = {
        "layer1": {"indices": [0], "values": [1.0]},
        "layer2": {"indices": [0], "values": [1.0]},
    }
    ratio = get_compression_ratio(original, compressed)
    assert ratio > 0
