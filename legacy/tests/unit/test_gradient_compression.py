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
    # Define dummy classes for test collection
    PowerSGDCompressor = None
    TopKCompressor = None
    ZstdCompressor = None


@pytest.mark.skipif(not HAS_GRADIENT_COMPRESSION, reason="gradient compression not available")
def test_powersgd_compressor_import():
    assert PowerSGDCompressor is not None


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_powersgd_creation():
    compressor = PowerSGDCompressor(rank=2, num_iterations=3)
    assert compressor.rank == 2
    assert compressor.num_iterations == 3


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_powersgd_compress_decompress():
    compressor = PowerSGDCompressor(rank=2)
    gradients = {
        "layer1.weight": torch.randn(10, 10),
        "layer2.weight": torch.randn(5, 5),
    }
    compressed = compressor.compress(gradients)
    assert "layer1.weight" in compressed
    assert "P" in compressed["layer1.weight"]
    assert "Q" in compressed["layer1.weight"]
    decompressed = compressor.decompress(compressed)
    assert "layer1.weight" in decompressed
    assert decompressed["layer1.weight"].shape == (10, 10)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_powersqd_error_feedback():
    compressor = PowerSGDCompressor(rank=2, use_error_feedback=True)
    grad1 = torch.randn(10, 10)
    compressor.compress({"layer1": grad1})
    grad2 = torch.randn(10, 10)
    compressor.compress({"layer1": grad2})
    assert len(compressor.error_feedback) > 0


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_topk_compressor():
    compressor = TopKCompressor(sparsity=0.9)
    grad = torch.randn(100)
    compressed = compressor.compress({"layer1": grad})
    assert "layer1" in compressed
    indices, values = compressed["layer1"]
    assert isinstance(indices, list)
    assert isinstance(values, list)
    assert len(indices) == len(values)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_topk_decompress():
    compressor = TopKCompressor(sparsity=0.9)
    grad = torch.randn(100)
    compressed = compressor.compress({"layer1": grad})
    decompressed = compressor.decompress(compressed, {"layer1": (100,)})
    assert "layer1" in decompressed
    assert decompressed["layer1"].shape == (100,)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_onebit_compressor():
    compressor = OneBitAdamCompressor()
    grad = torch.randn(100)
    compressed = compressor.compress({"layer1": grad})
    assert "layer1" in compressed
    assert "sign" in compressed["layer1"]
    assert "magnitude" in compressed["layer1"]


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_onebit_decompress():
    compressor = OneBitAdamCompressor()
    grad = torch.randn(100)
    compressed = compressor.compress({"layer1": grad})
    decompressed = compressor.decompress(compressed)
    assert "layer1" in decompressed
    assert decompressed["layer1"].shape == (100,)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_compression_ratio():
    compressor = PowerSGDCompressor(rank=2)
    gradients = {"layer1": torch.randn(100, 100)}
    compressed = compressor.compress(gradients)
    ratio = get_compression_ratio(gradients, compressed)
    assert ratio > 1.0


@pytest.mark.skipif(not HAS_GRADIENT_COMPRESSION, reason="gradient compression not available")
def test_zstd_compressor_import():
    assert ZstdCompressor is not None


@pytest.mark.skipif(not HAS_GRADIENT_COMPRESSION, reason="gradient compression not available")
def test_zstd_compressor_creation():
    compressor = ZstdCompressor(level=3)
    assert compressor.level == 3


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_zstd_compress_decompress():
    compressor = ZstdCompressor(level=3)
    original = torch.randn(100)
    compressed = compressor.compress_tensor(original)
    decompressed = compressor.decompress_tensor(
        compressed,
        shape=(100,),
        dtype=torch.float32,
    )
    assert decompressed.shape == original.shape
    assert torch.allclose(original, decompressed, atol=1e-5)


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_zstd_compression_ratio():
    compressor = ZstdCompressor(level=3)
    original = torch.zeros(1000)
    compressed = compressor.compress_tensor(original)
    ratio = compressor.get_compression_ratio(original, compressed)
    assert ratio >= 1.0


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_adaptive_compressor():
    compressor = AdaptiveCompression(threshold_small=500, threshold_large=50_000)
    large_grad = torch.randn(250, 250)
    compressed = compressor.compress({"layer1": large_grad})
    assert "layer1" in compressed
    assert compressed["layer1"].get("method") == "powersgd"


@pytest.mark.skipif(
    not HAS_TORCH or not HAS_GRADIENT_COMPRESSION,
    reason="torch or gradient compression not available",
)
def test_deep_gradient_compression_roundtrip():
    dgc = DeepGradientCompression(sparsity=0.95)
    g = torch.randn(32)
    c = dgc.compress({"w": g})
    out = dgc.decompress(c)
    assert "w" in out
    assert out["w"].shape == (32,)
