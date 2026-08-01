"""
Gradient Compression for DistribAI

Implements efficient gradient compression for distributed training:
- PowerSGD: Low-rank gradient approximation
- Top-K Sparsification: Keep only largest gradients
- Zstd compression: Lossless compression for transmission
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    import zstd

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False


class PowerSGDCompressor:
    """
    PowerSGD compressor using low-rank matrix approximation.

    As specified in RFC 002, PowerSGD is the best method for SLM training with:
    - Low-rank approximation of gradient matrices
    - Power iteration for refinement
    - Error feedback for convergence
    - Adaptive rank selection

    Attributes:
        rank: Rank for low-rank approximation
        num_iterations: Number of power iterations per compression
        use_error_feedback: Whether to accumulate approximation error
        error_feedback: Dictionary of accumulated errors per parameter
        P: Dictionary of left projection matrices
        Q: Dictionary of right projection matrices

    Example:
        compressor = PowerSGDCompressor(rank=2, num_iterations=3)
        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed)
    """

    def __init__(self, rank: int = 2, num_iterations: int = 3, use_error_feedback: bool = True):
        """
        Initialize the PowerSGD compressor.

        Args:
            rank: Rank for low-rank approximation (1-4 for SLMs)
            num_iterations: Number of power iterations per compression
            use_error_feedback: Whether to accumulate approximation error

        Example:
            >>> compressor = PowerSGDCompressor(rank=2, num_iterations=3)
        """
        self.rank = rank
        self.num_iterations = num_iterations
        self.use_error_feedback = use_error_feedback
        self.error_feedback: dict[str, torch.Tensor] = {}
        self.P: dict[str, torch.Tensor] = {}
        self.Q: dict[str, torch.Tensor] = {}

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, dict]:
        """
        Compress gradients using PowerSGD.

        Args:
            gradients: Dictionary of parameter name -> gradient tensor

        Returns:
            Compressed gradients as dict of param -> (P, Q) factors

        Example:
            >>> compressed = compressor.compress({"layer1.weight": grad_tensor})
        """
        compressed = {}
        for name, grad in gradients.items():
            if grad is None or grad.numel() == 0:
                continue
            if self.use_error_feedback and name in self.error_feedback:
                grad = grad + self.error_feedback[name]
            original_shape = grad.shape
            if grad.dim() == 0:
                grad = grad.view(1, 1)
            elif grad.dim() == 1:
                grad = grad.view(1, -1)
            elif grad.dim() > 2:
                grad = grad.view(grad.shape[0], -1)
            d, m = grad.shape
            rank = min(self.rank, min(d, m))
            if name not in self.P or self.P[name].shape != (d, rank):
                self.P[name] = torch.randn(d, rank, device=grad.device, dtype=grad.dtype) / np.sqrt(
                    d
                )
                self.Q[name] = torch.randn(m, rank, device=grad.device, dtype=grad.dtype) / np.sqrt(
                    m
                )
            P = self.P[name]
            Q = self.Q[name]
            for _ in range(self.num_iterations):
                Q = torch.matmul(grad.t(), P)
                Q = self._orthogonalize(Q)
                P = torch.matmul(grad, Q)
                P = self._orthogonalize(P)
            self.P[name] = P
            self.Q[name] = Q
            if self.use_error_feedback:
                approx = torch.matmul(P, Q.t())
                self.error_feedback[name] = grad - approx
            compressed[name] = {"P": P, "Q": Q, "shape": original_shape, "method": "powersgd"}
        return compressed

    def decompress(self, compressed: dict[str, dict]) -> dict[str, torch.Tensor]:
        """
        Decompress PowerSGD-compressed gradients.

        Args:
            compressed: Compressed gradients from compress()

        Returns:
            Reconstructed gradient tensors

        Example:
            >>> decompressed = compressor.decompress(compressed)
        """
        decompressed = {}
        for name, data in compressed.items():
            if data.get("method") != "powersgd":
                continue
            P = data["P"]
            Q = data["Q"]
            original_shape = data["shape"]
            grad_approx = torch.matmul(P, Q.t())
            if len(original_shape) > 2:
                grad_approx = grad_approx.view(original_shape)
            decompressed[name] = grad_approx
        return decompressed

    def _orthogonalize(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Orthogonalize matrix using QR decomposition.

        Args:
            matrix: Matrix to orthogonalize

        Returns:
            Orthogonalized matrix
        """
        q, _ = torch.linalg.qr(matrix)
        return q

    def reset(self) -> None:
        """
        Reset the compressor state.

        Clears error feedback and projection matrices.

        Example:
            >>> compressor.reset()
        """
        self.error_feedback.clear()
        self.P.clear()
        self.Q.clear()


class TopKCompressor:
    """
    Top-K sparsification compressor.

    Keeps only the largest gradient values to reduce communication overhead.
    Maintains momentum buffer for error feedback.

    Attributes:
        sparsity: Fraction of gradients to prune (0.0-1.0)
        momentum_buffer: Momentum buffer for error feedback

    Example:
        compressor = TopKCompressor(sparsity=0.99)
        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed)
    """

    def __init__(self, sparsity: float = 0.99):
        """
        Initialize the Top-K compressor.

        Args:
            sparsity: Fraction of gradients to prune (0.0-1.0)

        Example:
            >>> compressor = TopKCompressor(sparsity=0.99)
        """
        self.sparsity = sparsity
        self.momentum_buffer: dict[str, torch.Tensor] = {}

    def compress(
        self, gradients: dict[str, torch.Tensor]
    ) -> dict[str, tuple[list[int], list[float]]]:
        """
        Compress gradients using Top-K sparsification.

        Args:
            gradients: Dictionary of parameter name -> gradient tensor

        Returns:
            Compressed gradients as dict of param -> (indices, values)

        Example:
            >>> compressed = compressor.compress({"layer1.weight": grad_tensor})
        """
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            grad_flat = grad.flatten().cpu().numpy()
            k = max(1, int(len(grad_flat) * (1 - self.sparsity)))
            indices = np.argpartition(np.abs(grad_flat), -k)[-k:]
            indices = indices[np.argsort(indices)]
            values = grad_flat[indices].tolist()
            compressed[name] = (indices.tolist(), values)
            if name not in self.momentum_buffer:
                self.momentum_buffer[name] = torch.zeros_like(grad)
            mask = torch.zeros_like(grad)
            mask.view(-1)[indices] = 1
            self.momentum_buffer[name] += grad * (1 - mask)
        return compressed

    def decompress(
        self,
        compressed: dict[str, tuple[list[int], list[float]]],
        shape: dict[str, tuple[int, ...]],
    ) -> dict[str, torch.Tensor]:
        """
        Decompress gradients.

        Args:
            compressed: Compressed gradients from compress()
            shape: Dictionary of parameter name -> tensor shape

        Returns:
            Reconstructed gradient tensors

        Example:
            >>> decompressed = compressor.decompress(compressed, shape_dict)
        """
        decompressed = {}
        for name, (indices, values) in compressed.items():
            if name not in shape:
                continue
            grad = torch.zeros(np.prod(shape[name]), dtype=torch.float32)
            grad[indices] = torch.tensor(values, dtype=torch.float32)
            if name in self.momentum_buffer:
                grad += self.momentum_buffer[name].flatten()
            decompressed[name] = grad.reshape(shape[name])
        return decompressed

    def reset_momentum(self) -> None:
        """
        Clear the momentum buffer.

        Example:
            >>> compressor.reset_momentum()
        """
        self.momentum_buffer.clear()


class QuantizationCompressor:
    """
    Quantization compressor for gradient compression.

    Reduces precision of gradients to reduce bandwidth requirements.

    Attributes:
        bits: Number of bits for quantization
        scale: Scale factor for quantization

    Example:
        compressor = QuantizationCompressor(bits=8)
        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed, shape_dict)
    """

    def __init__(self, bits: int = 8):
        """
        Initialize the quantization compressor.

        Args:
            bits: Number of bits for quantization (4, 8, or 16)

        Example:
            >>> compressor = QuantizationCompressor(bits=8)
        """
        self.bits = bits
        self.scale = 2 ** (bits - 1) - 1

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, tuple[list[int], float]]:
        """
        Compress gradients using uniform quantization.
        Args:
            gradients: Dictionary of parameter name -> gradient tensor
        Returns:
            Compressed gradients as dict of param -> (quantized_values, scale_factor)
        """
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            grad_flat = grad.flatten().cpu().numpy()
            min_val = np.min(grad_flat)
            max_val = np.max(grad_flat)
            if max_val == min_val:
                scale_factor = 1.0
            else:
                scale_factor = (max_val - min_val) / self.scale
            quantized = np.round((grad_flat - min_val) / scale_factor).astype(np.int16)
            quantized = np.clip(quantized, 0, self.scale)
            compressed[name] = (quantized.tolist(), scale_factor, min_val)
        return compressed

    def decompress(
        self,
        compressed: dict[str, tuple[list[int], float, float]],
        shape: dict[str, tuple[int, ...]],
    ) -> dict[str, torch.Tensor]:
        """
        Decompress quantized gradients.
        Args:
            compressed: Compressed gradients from compress()
            shape: Dictionary of parameter name -> tensor shape
        Returns:
            Reconstructed gradient tensors
        """
        decompressed = {}
        for name, (quantized, scale_factor, min_val) in compressed.items():
            if name not in shape:
                continue
            grad_flat = np.array(quantized) * scale_factor + min_val
            decompressed[name] = torch.tensor(grad_flat, dtype=torch.float32).reshape(shape[name])
        return decompressed


class HybridCompressor:
    def __init__(self, sparsity: float = 0.95, bits: int = 8):
        """
        Initialize hybrid compressor.
        Args:
            sparsity: Fraction of gradients to prune
            bits: Number of bits for quantization
        """
        self.topk = TopKCompressor(sparsity=sparsity)
        self.quant = QuantizationCompressor(bits=bits)

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, dict]:
        """
        Compress gradients using hybrid approach.
        Args:
            gradients: Dictionary of parameter name -> gradient tensor
        Returns:
            Compressed gradients with metadata
        """
        sparse_grads = self.topk.compress(gradients)
        compressed = {}
        for name, (indices, values) in sparse_grads.items():
            values_array = np.array(values)
            min_val = np.min(values_array)
            max_val = np.max(values_array)
            if max_val == min_val:
                scale_factor = 1.0
            else:
                scale_factor = (max_val - min_val) / (2**self.quant.bits - 1)
            quantized = np.round((values_array - min_val) / scale_factor).astype(np.int16)
            quantized = np.clip(quantized, 0, 2**self.quant.bits - 1)
            compressed[name] = {
                "indices": indices,
                "values": quantized.tolist(),
                "scale": scale_factor,
                "min_val": min_val,
                "shape": gradients[name].shape,
            }
        return compressed

    def decompress(self, compressed: dict[str, dict]) -> dict[str, torch.Tensor]:
        """
        Decompress hybrid-compressed gradients.
        Args:
            compressed: Compressed gradients from compress()
        Returns:
            Reconstructed gradient tensors
        """
        decompressed = {}
        for name, data in compressed.items():
            values = np.array(data["values"]) * data["scale"] + data["min_val"]
            grad = torch.zeros(np.prod(data["shape"]), dtype=torch.float32)
            grad[data["indices"]] = torch.tensor(values, dtype=torch.float32)
            if name in self.topk.momentum_buffer:
                grad += self.topk.momentum_buffer[name].flatten()
            decompressed[name] = grad.reshape(data["shape"])
        return decompressed

    def reset_momentum(self):
        self.topk.reset_momentum()


class OneBitAdamCompressor:
    """
    1-bit Adam compressor for extreme compression.
    As specified in RFC 002, 1-bit Adam provides 32x compression with Adam-like convergence.
    """

    def __init__(self, use_quantization: bool = True):
        """
        Initialize 1-bit Adam compressor.
        Args:
            use_quantization: Whether to use 1-bit quantization
        """
        self.use_quantization = use_quantization
        self.momentums: dict[str, torch.Tensor] = {}

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, dict]:
        """
        Compress gradients using 1-bit quantization.
        Args:
            gradients: Dictionary of parameter name -> gradient tensor
        Returns:
            Compressed gradients with metadata
        """
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            sign = torch.sign(grad)
            magnitude = torch.abs(grad).mean()
            compressed[name] = {
                "sign": sign,
                "magnitude": magnitude.item(),
                "shape": grad.shape,
                "method": "onebit",
            }
        return compressed

    def decompress(self, compressed: dict[str, dict]) -> dict[str, torch.Tensor]:
        """
        Decompress 1-bit compressed gradients.
        Args:
            compressed: Compressed gradients from compress()
        Returns:
            Reconstructed gradient tensors
        """
        decompressed = {}
        for name, data in compressed.items():
            if data.get("method") != "onebit":
                continue
            sign = data["sign"]
            magnitude = data["magnitude"]
            shape = data["shape"]
            grad = sign.float() * magnitude
            decompressed[name] = grad.reshape(shape)
        return decompressed


class DeepGradientCompression:
    """
    Deep Gradient Compression (DGC) using Top-K sparsification with momentum correction.
    This is the production implementation that wraps TopKCompressor with a dict-based
    interface compatible with AdaptiveCompression.
    """

    def __init__(self, sparsity: float = 0.99):
        """
        Initialize DGC compressor.
        Args:
            sparsity: Fraction of gradients to prune (0.0-1.0)
        """
        self._topk = TopKCompressor(sparsity=sparsity)

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, dict]:
        """
        Compress gradients using Top-K sparsification.
        Args:
            gradients: Dictionary of parameter name -> gradient tensor
        Returns:
            Compressed gradients with method metadata
        """
        tuple_compressed = self._topk.compress(gradients)
        compressed = {}
        for name, (indices, values) in tuple_compressed.items():
            compressed[name] = {
                "indices": indices,
                "values": values,
                "method": "dgc",
                "shape": gradients[name].shape,
            }
        return compressed

    def decompress(self, compressed: dict[str, dict]) -> dict[str, torch.Tensor]:
        """
        Decompress DGC-compressed gradients.
        Args:
            compressed: Compressed gradients from compress()
        Returns:
            Reconstructed gradient tensors
        """
        tuple_compressed = {}
        shapes = {}
        for name, data in compressed.items():
            tuple_compressed[name] = (data["indices"], data["values"])
            shapes[name] = data["shape"]
        return self._topk.decompress(tuple_compressed, shapes)

    def reset_momentum(self):
        self._topk.reset_momentum()


class AdaptiveCompression:
    """
    Adaptive compression that automatically selects the best method.
    Combines PowerSGD, DGC, and 1-bit Adam based on gradient characteristics.
    """

    def __init__(self, threshold_small: float = 1000, threshold_large: float = 100000):
        """
        Initialize adaptive compressor.
        Args:
            threshold_small: Size threshold for using 1-bit Adam
            threshold_large: Size threshold for using PowerSGD
        """
        self.threshold_small = threshold_small
        self.threshold_large = threshold_large
        self.powersgd = PowerSGDCompressor(rank=2, num_iterations=3)
        self.dgc = DeepGradientCompression(sparsity=0.99)
        self.onebit = OneBitAdamCompressor()
        self.method_used: dict[str, str] = {}

    def compress(self, gradients: dict[str, torch.Tensor]) -> dict[str, dict]:
        """
        Compress gradients using adaptive method selection.
        Args:
            gradients: Dictionary of parameter name -> gradient tensor
        Returns:
            Compressed gradients with method metadata
        """
        compressed = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            size = grad.numel()
            if size < self.threshold_small:
                result = self.onebit.compress({name: grad})[name]
                self.method_used[name] = "onebit"
            elif size > self.threshold_large:
                result = self.powersgd.compress({name: grad})[name]
                self.method_used[name] = "powersgd"
            else:
                dgc_result = self.dgc.compress({name: grad})
                result = dgc_result[name]
                self.method_used[name] = "dgc"
            compressed[name] = result
        return compressed

    def decompress(self, compressed: dict[str, dict]) -> dict[str, torch.Tensor]:
        """
        Decompress adaptive-compressed gradients.
        Args:
            compressed: Compressed gradients from compress()
        Returns:
            Reconstructed gradient tensors
        """
        decompressed = {}
        powersgd_compressed = {k: v for k, v in compressed.items() if v.get("method") == "powersgd"}
        dgc_compressed = {k: v for k, v in compressed.items() if v.get("method") == "dgc"}
        onebit_compressed = {k: v for k, v in compressed.items() if v.get("method") == "onebit"}
        if powersgd_compressed:
            decompressed.update(self.powersgd.decompress(powersgd_compressed))
        if dgc_compressed:
            decompressed.update(self.dgc.decompress(dgc_compressed))
        if onebit_compressed:
            decompressed.update(self.onebit.decompress(onebit_compressed))
        return decompressed

    def reset(self):
        self.powersgd.reset()
        self.dgc.reset_momentum()
        self.method_used.clear()


def get_compression_ratio(original: dict[str, torch.Tensor], compressed: dict) -> float:
    """
    Calculate compression ratio.
    Args:
        original: Original gradients
        compressed: Compressed gradients
    Returns:
        Compression ratio (original_size / compressed_size)
    """
    original_size = sum(grad.numel() * 4 for grad in original.values())
    compressed_size = 0
    for data in compressed.values():
        method = data.get("method")
        if method == "powersgd":
            P = data.get("P")
            Q = data.get("Q")
            if P is not None and Q is not None:
                compressed_size += P.numel() * 4 + Q.numel() * 4
        elif method == "dgc":
            compressed_size += len(data["indices"]) * 4 + len(data["values"]) * 4
        elif method == "onebit":
            compressed_size += data["sign"].numel() // 8 + 4
        else:
            if "indices" in data and "values" in data:
                compressed_size += len(data["indices"]) * 4 + len(data["values"]) * 4
    return original_size / max(1, compressed_size)


class ZstdCompressor:
    """
    Zstd lossless compression for gradient transmission.
    Provides high-speed compression for network transmission.
    Can be combined with PowerSGD for maximum compression.
    """

    def __init__(self, level: int = 3):
        """
        Initialize Zstd compressor.
        Args:
            level: Compression level (1-22, higher = better compression but slower)
        """
        self.level = level
        self.enabled = ZSTD_AVAILABLE
        if not self.enabled:
            logging.warning("zstd not installed. Zstd compression disabled.")

    def compress_tensor(self, tensor: torch.Tensor) -> bytes:
        """
        Compress a tensor using Zstd.
        Args:
            tensor: PyTorch tensor to compress
        Returns:
            Compressed bytes
        """
        if not self.enabled:
            return tensor.cpu().numpy().tobytes()
        np_array = tensor.cpu().numpy()
        raw_bytes = np_array.tobytes()
        compressed = zstd.compress(raw_bytes, self.level)
        return compressed

    def decompress_tensor(
        self, compressed: bytes, shape: tuple[int, ...], dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Decompress bytes back to tensor.
        Args:
            compressed: Compressed bytes
            shape: Original tensor shape
            dtype: Original tensor dtype
        Returns:
            Decompressed tensor
        """
        if not self.enabled:
            import numpy as np

            np_dtype = {
                torch.float32: np.float32,
                torch.float16: np.float16,
                torch.int32: np.int32,
                torch.int64: np.int64,
            }.get(dtype, np.float32)
            np_array = np.frombuffer(compressed, dtype=np_dtype).reshape(shape)
            return torch.from_numpy(np_array.copy())
        raw_bytes = zstd.decompress(compressed)
        import numpy as np

        np_dtype = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.int32: np.int32,
            torch.int64: np.int64,
        }.get(dtype, np.float32)
        np_array = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(shape).copy()
        tensor = torch.from_numpy(np_array)
        return tensor

    def get_compression_ratio(self, original: torch.Tensor, compressed: bytes) -> float:
        original_bytes = original.numel() * 4
        compressed_bytes = len(compressed)
        return original_bytes / max(1, compressed_bytes)
