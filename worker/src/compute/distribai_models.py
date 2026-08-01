"""Registered DistribAI nn.Module families plus declarative architecture builders."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

try:
    from services_python.architecture_config import validate_architecture_config
except ImportError:  # pragma: no cover - standalone worker import fallback
    _FALLBACK_FAMILIES = {
        "decoder_transformer",
        "gru",
        "gated_conv",
        "moe_decoder",
        "lstm",
        "resnet_lm",
        "hybrid_attn_rnn",
        "dense_ffn",
    }
    _FALLBACK_KEYS = {
        "version",
        "family",
        "architecture",
        "dim",
        "n_unique_layers",
        "n_logical_layers",
        "n_heads",
        "n_kv_heads",
        "ffn_dim",
        "dropout",
        "seq_len",
        "sliding_window",
        "engram_dim",
        "mhc_expansion",
        "num_experts",
        "top_k",
        "conv_kernel",
        "gru_layers",
        "attn_res_block_size",
        "qk_norm",
        "use_head_gating",
        "embedding_scale",
        "grad_checkpoint",
        "mtp_horizons",
    }
    _FALLBACK_MAX_ESTIMATED_PARAMETERS = 512_000_000
    _FALLBACK_MAX_TRANSFORMER_SEQ_LEN = 8192
    _FALLBACK_LIMITS = {
        "dim": (16, 4096),
        "n_unique_layers": (1, 64),
        "n_logical_layers": (1, 128),
        "n_heads": (1, 64),
        "n_kv_heads": (1, 64),
        "ffn_dim": (16, 16384),
        "seq_len": (8, 32768),
        "num_experts": (1, 64),
        "top_k": (1, 16),
        "conv_kernel": (2, 31),
        "gru_layers": (1, 16),
        "sliding_window": (0, 32768),
        "engram_dim": (0, 1024),
        "mhc_expansion": (1, 16),
        "attn_res_block_size": (0, 64),
    }
    _FALLBACK_BOOL_KEYS = frozenset(
        {"qk_norm", "use_head_gating", "embedding_scale", "grad_checkpoint"}
    )
    _FALLBACK_ATTENTION_FAMILIES = frozenset(
        {"decoder_transformer", "hybrid_attn_rnn", "moe_decoder"}
    )
    _FALLBACK_DECODER_ONLY_KEYS = frozenset(
        {"engram_dim", "mhc_expansion", "mtp_horizons", "grad_checkpoint", "attn_res_block_size"}
    )

    def _fallback_depth(value: Any) -> int:
        """Iterative nest-depth so adversarial JSON cannot blow the call stack."""
        maximum = 0
        stack = [(value, 0)]
        while stack:
            current, depth = stack.pop()
            if not isinstance(current, (dict, list)):
                continue
            maximum = max(maximum, depth)
            children = current.values() if isinstance(current, dict) else current
            stack.extend((child, depth + 1) for child in children)
        return maximum

    def validate_architecture_config(value: dict[str, Any]) -> dict[str, Any]:
        """Mirror orchestrator bounds when the worker is imported without services_python."""
        if not isinstance(value, dict):
            raise ValueError("architecture_config must be an object")
        if _fallback_depth(value) > 6:
            raise ValueError("architecture_config is nested too deeply")
        try:
            encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        except (RecursionError, TypeError, ValueError) as exc:
            raise ValueError("architecture_config must contain JSON-finite values") from exc
        if len(encoded) > 64 * 1024:
            raise ValueError("architecture_config exceeds 64 KiB")
        unknown = set(value) - _FALLBACK_KEYS
        if unknown:
            raise ValueError(f"unsupported architecture_config keys: {sorted(unknown)}")
        normalized = dict(value)
        if "family" in normalized and "architecture" in normalized:
            raw_family = normalized["family"]
            raw_architecture = normalized["architecture"]
            if (
                not isinstance(raw_family, str)
                or not isinstance(raw_architecture, str)
                or raw_family.strip().lower() != raw_architecture.strip().lower()
            ):
                raise ValueError("architecture_config.family and architecture must agree")
        family = normalized.get("family", normalized.get("architecture"))
        if not isinstance(family, str) or family.strip().lower() not in _FALLBACK_FAMILIES:
            raise ValueError("architecture_config.family is unsupported")
        normalized["family"] = family.strip().lower()
        normalized["architecture"] = normalized["family"]
        raw_version = normalized.get("version", 1)
        if isinstance(raw_version, bool) or not isinstance(raw_version, int):
            raise ValueError("architecture_config.version must be an integer")
        normalized["version"] = raw_version
        if normalized["version"] != 1:
            raise ValueError("unsupported architecture_config version")
        for key, (minimum, maximum) in _FALLBACK_LIMITS.items():
            if key in normalized:
                item = normalized[key]
                if isinstance(item, bool) or not isinstance(item, int):
                    raise ValueError(f"architecture_config.{key} must be an integer")
                if not minimum <= item <= maximum:
                    raise ValueError(f"architecture_config.{key} is out of bounds")
        dropout = normalized.get("dropout")
        if dropout is not None and (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not math.isfinite(float(dropout))
            or not 0 <= float(dropout) <= 0.5
        ):
            raise ValueError("architecture_config.dropout is out of bounds")
        if dropout is not None:
            normalized["dropout"] = float(dropout)
        for key in _FALLBACK_BOOL_KEYS:
            if key in normalized and not isinstance(normalized[key], bool):
                raise ValueError(f"architecture_config.{key} must be a boolean")
        if "mtp_horizons" in normalized:
            horizons = normalized["mtp_horizons"]
            if (
                not isinstance(horizons, list)
                or not horizons
                or len(horizons) > 4
                or any(isinstance(h, bool) or not isinstance(h, int) for h in horizons)
                or any(not 1 <= h <= 8 for h in horizons)
                or len(set(horizons)) != len(horizons)
            ):
                raise ValueError(
                    "architecture_config.mtp_horizons must be a non-empty list of at most "
                    "4 unique integers between 1 and 8"
                )
            normalized["mtp_horizons"] = sorted(horizons)
        if normalized["family"] not in _FALLBACK_ATTENTION_FAMILIES:
            if int(normalized.get("sliding_window", 0)):
                raise ValueError(
                    "architecture_config.sliding_window requires an attention family"
                )
            for key in ("qk_norm", "use_head_gating"):
                if normalized.get(key) is True:
                    raise ValueError(f"architecture_config.{key} requires an attention family")
        if normalized["family"] != "decoder_transformer":
            for key in _FALLBACK_DECODER_ONLY_KEYS:
                if key not in normalized:
                    continue
                value_for_key = normalized[key]
                is_off = (
                    value_for_key in (0, False)
                    or (key == "mhc_expansion" and value_for_key == 1)
                    or (key == "mtp_horizons" and value_for_key == [1])
                )
                if not is_off:
                    raise ValueError(
                        f"architecture_config.{key} is only supported by decoder_transformer"
                    )
        elif int(normalized.get("attn_res_block_size", 0)) > 0 and int(
            normalized.get("mhc_expansion", 1)
        ) > 1:
            raise ValueError(
                "architecture_config.attn_res_block_size cannot be combined with mhc_expansion > 1"
            )
        if normalized["family"] in _FALLBACK_ATTENTION_FAMILIES:
            dim = int(normalized.get("dim", 256))
            heads = int(normalized.get("n_heads", 8))
            if dim % heads:
                raise ValueError("architecture_config.dim must be divisible by n_heads")
            kv_heads = int(normalized.get("n_kv_heads", heads))
            if heads % kv_heads:
                raise ValueError("architecture_config.n_heads must be divisible by n_kv_heads")
            if int(normalized.get("seq_len", 512)) > _FALLBACK_MAX_TRANSFORMER_SEQ_LEN:
                raise ValueError("architecture_config.seq_len is too large for transformer attention")
            if normalized["family"] == "decoder_transformer" and int(
                normalized.get("n_logical_layers", normalized.get("n_unique_layers", 8))
            ) < int(normalized.get("n_unique_layers", 8)):
                raise ValueError("n_logical_layers cannot be less than n_unique_layers")
            window = int(normalized.get("sliding_window", 0))
            if window and window >= int(normalized.get("seq_len", 512)):
                normalized["sliding_window"] = 0
        if normalized["family"] == "moe_decoder" and int(normalized.get("top_k", 2)) > int(
            normalized.get("num_experts", 4)
        ):
            raise ValueError("architecture_config.top_k cannot exceed num_experts")
        dim = int(normalized.get("dim", 256))
        ffn_dim = int(normalized.get("ffn_dim", 4 * dim))
        layers = int(normalized.get("n_unique_layers", normalized.get("n_logical_layers", 8)))
        if normalized["family"] == "decoder_transformer":
            estimate = layers * (4 * dim * dim + 2 * dim * ffn_dim) + 512 * dim
            engram_dim = int(normalized.get("engram_dim", 0))
            if engram_dim:
                estimate += 4096 * engram_dim + engram_dim * dim
            estimate += sum(
                256 * dim for h in (normalized.get("mtp_horizons") or []) if int(h) > 1
            )
        elif normalized["family"] == "gru":
            gru_layers = int(normalized.get("gru_layers", 2))
            estimate = gru_layers * 3 * (2 * dim * dim + 2 * dim) + 512 * dim
        elif normalized["family"] == "lstm":
            lstm_layers = int(normalized.get("gru_layers", 2))
            estimate = lstm_layers * 4 * (2 * dim * dim + 2 * dim) + 512 * dim
        elif normalized["family"] == "gated_conv":
            conv_layers = int(normalized.get("n_logical_layers", 6))
            estimate = conv_layers * 2 * dim * dim * int(normalized.get("conv_kernel", 5)) + 512 * dim
        elif normalized["family"] == "resnet_lm":
            res_layers = int(normalized.get("n_logical_layers", 6))
            estimate = (
                res_layers * (dim * dim * int(normalized.get("conv_kernel", 5)) + dim * dim) + 512 * dim
            )
        elif normalized["family"] == "hybrid_attn_rnn":
            hybrid_layers = int(normalized.get("n_logical_layers", 8))
            attn_layers = (hybrid_layers + 1) // 2
            rnn_layers = hybrid_layers // 2
            estimate = (
                attn_layers * (4 * dim * dim + 2 * dim * ffn_dim)
                + rnn_layers * 3 * (2 * dim * dim + 2 * dim)
                + 512 * dim
            )
        elif normalized["family"] == "dense_ffn":
            ffn_layers = int(normalized.get("n_logical_layers", 8))
            estimate = ffn_layers * (2 * dim * ffn_dim) + 512 * dim
        else:
            moe_layers = int(normalized.get("n_logical_layers", 8))
            experts = int(normalized.get("num_experts", 4))
            estimate = (
                moe_layers * (4 * dim * dim + experts * 2 * dim * ffn_dim + dim * experts)
                + 512 * dim
            )
        if estimate > _FALLBACK_MAX_ESTIMATED_PARAMETERS:
            raise ValueError("architecture_config estimated parameter count is too large")
        return normalized

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Shared knobs for every registered native language-model family.

    Feature knobs default to "off" so uploaded configs opt in explicitly:
    ``sliding_window=0`` keeps full causal attention, ``engram_dim=0`` skips
    the hashed n-gram memory, ``mhc_expansion=1`` keeps a single residual
    stream, ``mtp_horizons=[1]`` trains only the standard next-token head,
    and ``grad_checkpoint=False`` keeps activations resident.
    """

    family: str = "decoder_transformer"
    dim: int = 64
    n_unique_layers: int = 4
    n_logical_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.0
    seq_len: int = 512
    mtp_horizons: list[int] = field(default_factory=lambda: [1])
    grad_checkpoint: bool = False
    sliding_window: int = 0
    engram_dim: int = 0
    mhc_expansion: int = 1
    num_experts: int = 4
    top_k: int = 2
    conv_kernel: int = 5
    gru_layers: int = 2
    attn_res_block_size: int = 0
    qk_norm: bool = False
    use_head_gating: bool = False
    embedding_scale: bool = False


class _RMSNorm(nn.Module):
    """Lightweight RMSNorm used for optional Q/K normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * inputs * torch.rsqrt(variance + self.eps)


def build_causal_mask(seq_len: int, sliding_window: int = 0) -> torch.Tensor:
    """Additive float attention mask: causal, optionally banded.

    Args:
        seq_len: Mask side length.
        sliding_window: When > 0, each query may attend to at most this many
            most recent positions (itself included); older keys get ``-inf``.
            0 keeps full causal attention.

    Returns:
        ``(seq_len, seq_len)`` float tensor of 0 / ``-inf`` entries suitable
        for both the custom GQA blocks and ``nn.TransformerEncoderLayer``.
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    if sliding_window > 0:
        # Ban keys further back than the window: query i sees keys (i-w, i].
        mask = mask + torch.tril(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=-sliding_window
        )
    return mask


class _EngramMemory(nn.Module):
    """Hashed bigram "engram" memory added to the token embedding stream.

    A small learned table is addressed by a deterministic hash of each
    ``(previous_token, current_token)`` pair, giving the model O(1) access to
    local n-gram statistics without spending attention capacity on them. The
    lookup at position ``t`` only touches tokens ``t-1`` and ``t``, so
    causality for next-token prediction is preserved.
    """

    N_SLOTS = 4096

    def __init__(self, vocab_size: int, dim: int, engram_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.table = nn.Embedding(self.N_SLOTS, engram_dim)
        self.proj = nn.Linear(engram_dim, dim, bias=False)
        # Start as a no-op so enabling engrams never destabilizes early steps.
        nn.init.zeros_(self.proj.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        previous = torch.nn.functional.pad(tokens[:, :-1], (1, 0), value=0)
        # 31 is coprime with the power-of-two table so bigrams spread evenly.
        slots = (previous * 31 + tokens) % self.N_SLOTS
        return self.proj(self.table(slots))


class _HyperConnections(nn.Module):
    """Static hyper-connections: ``rate`` parallel residual streams per layer.

    Each block reads a learned mixture of the streams, and its delta
    (block output minus block input) is written back through learned
    per-stream weights while a learned matrix mixes the streams themselves.
    Initialization reproduces the plain single-stream residual network
    exactly (read/write focus on stream 0, identity mixing), so enabling the
    knob is loss-neutral at step 0.
    """

    def __init__(self, rate: int) -> None:
        super().__init__()
        self.rate = rate
        read = torch.zeros(rate)
        read[0] = 1.0
        self.read = nn.Parameter(read.clone())
        self.write = nn.Parameter(read.clone())
        self.mix = nn.Parameter(torch.eye(rate))

    def read_input(self, streams: torch.Tensor) -> torch.Tensor:
        """Blend streams ``(rate, B, T, D)`` into one block input ``(B, T, D)``."""
        return torch.einsum("r,rbtd->btd", self.read, streams)

    def write_output(self, streams: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Mix streams and add the block delta into each per its write weight."""
        mixed = torch.einsum("sr,rbtd->sbtd", self.mix, streams)
        return mixed + self.write.view(-1, 1, 1, 1) * delta.unsqueeze(0)


class _CausalSelfAttention(nn.Module):
    """Pre-norm causal self-attention with optional GQA, QK-norm, and gating.

    Unlike ``_GroupedQueryAttnBlock`` this module carries no feed-forward
    sublayer, so families that pair attention with their own mixer (MoE
    experts, recurrent blocks) can reuse it directly.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        qk_norm: bool = False,
        use_head_gating: bool = False,
    ) -> None:
        super().__init__()
        heads = n_heads if dim % max(1, n_heads) == 0 else 1
        kv_heads = int(n_kv_heads if n_kv_heads is not None else heads)
        kv_heads = max(1, min(kv_heads, heads))
        if heads % kv_heads:
            kv_heads = heads
        self.n_heads = heads
        self.n_kv_heads = kv_heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = _RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = _RMSNorm(self.head_dim) if qk_norm else None
        self.head_gate = nn.Linear(dim, heads, bias=True) if use_head_gating else None
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        x = self.norm(hidden)
        query = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        weights = self.attn_drop(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, value)
        if self.head_gate is not None:
            gates = torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
            context = context * gates
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return hidden + self.o_proj(context)


class _GroupedQueryAttnBlock(nn.Module):
    """Decoder block with optional GQA, QK-norm, and per-head gating."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        use_head_gating: bool = False,
    ) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if n_heads % max(1, n_kv_heads):
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = max(1, n_kv_heads)
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.use_head_gating = use_head_gating
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = _RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = _RMSNorm(self.head_dim) if qk_norm else None
        self.head_gate = nn.Linear(dim, n_heads, bias=True) if use_head_gating else None
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def _repeat_kv(self, values: torch.Tensor) -> torch.Tensor:
        if self.n_kv_heads == self.n_heads:
            return values
        repeats = self.n_heads // self.n_kv_heads
        return values.repeat_interleave(repeats, dim=1)

    def forward(self, hidden: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        residual = hidden
        x = self.norm1(hidden)
        query = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if src_mask is not None:
            scores = scores + src_mask.unsqueeze(0).unsqueeze(0)
        weights = self.attn_drop(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, value)
        if self.head_gate is not None:
            gates = torch.sigmoid(self.head_gate(x)).transpose(1, 2).unsqueeze(-1)
            context = context * gates
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        hidden = residual + self.o_proj(context)
        return hidden + self.ffn(self.norm2(hidden))


# Auxiliary multi-token-prediction heads are trained at a fraction of the main
# next-token loss so they shape representations without dominating them.
MTP_AUX_LOSS_WEIGHT = 0.3


class DistribAITinyLanguageModel(nn.Module):
    """Causal decoder transformer that can reuse unique layers across logical depth.

    Supported feature knobs (all opt-in, see ``ModelConfig``):

    - ``sliding_window``: banded causal attention (query sees at most the
      window's most recent keys).
    - ``engram_dim``: hashed bigram memory added to the embedding stream.
    - ``mhc_expansion``: static hyper-connections with that many parallel
      residual streams.
    - ``mtp_horizons``: auxiliary multi-token-prediction heads for horizons
      greater than one (only exercised through :meth:`compute_loss`).
    - ``grad_checkpoint``: activation checkpointing per logical layer during
      training to cut peak memory roughly by depth.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 64,
        n_unique_layers: int = 4,
        n_logical_layers: int | None = None,
        n_heads: int = 4,
        n_kv_heads: int | None = None,
        ffn_dim: int = 256,
        dropout: float = 0.0,
        seq_len: int = 2048,
        qk_norm: bool = False,
        use_head_gating: bool = False,
        embedding_scale: bool = False,
        attn_res_block_size: int = 0,
        sliding_window: int = 0,
        engram_dim: int = 0,
        mhc_expansion: int = 1,
        mtp_horizons: list[int] | None = None,
        grad_checkpoint: bool = False,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        heads = n_heads if dim % max(1, n_heads) == 0 else 1
        kv_heads = int(n_kv_heads if n_kv_heads is not None else heads)
        kv_heads = max(1, min(kv_heads, heads))
        if heads % kv_heads:
            kv_heads = heads
        unique_layer_count = max(1, int(n_unique_layers))
        logical_layer_count = max(
            unique_layer_count,
            int(n_logical_layers or unique_layer_count),
        )
        self.embedding_scale = bool(embedding_scale)
        self.attn_res_block_size = max(0, int(attn_res_block_size))
        self.sliding_window = max(0, int(sliding_window))
        self.grad_checkpoint = bool(grad_checkpoint)
        if self.attn_res_block_size and int(mhc_expansion) > 1:
            raise ValueError("attn_res_block_size cannot be combined with mhc_expansion > 1")
        self.embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(seq_len, dim)
        self.engram = _EngramMemory(vocab_size, dim, engram_dim) if int(engram_dim) > 0 else None
        self.register_buffer(
            "_causal_mask",
            build_causal_mask(seq_len, self.sliding_window),
            persistent=False,
        )
        use_custom_attn = (
            kv_heads != heads
            or qk_norm
            or use_head_gating
            or self.attn_res_block_size > 0
        )
        # Keep layers.* checkpoint keys stable. Logical depth points at unregistered
        # aliases so shared weights are not duplicated inside state_dict.
        if use_custom_attn:
            self.layers = nn.ModuleList(
                [
                    _GroupedQueryAttnBlock(
                        dim,
                        n_heads=heads,
                        n_kv_heads=kv_heads,
                        ffn_dim=ffn_dim,
                        dropout=dropout,
                        qk_norm=qk_norm,
                        use_head_gating=use_head_gating,
                    )
                    for _ in range(unique_layer_count)
                ]
            )
            self._stock_encoder = False
        else:
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        dim,
                        nhead=heads,
                        dim_feedforward=ffn_dim,
                        dropout=dropout,
                        batch_first=True,
                    )
                    for _ in range(unique_layer_count)
                ]
            )
            self._stock_encoder = True
        self.logical_layers = [
            self.layers[index % unique_layer_count] for index in range(logical_layer_count)
        ]
        # Hyper-connection mixers are cheap (rate + rate + rate^2 scalars) and
        # deliberately per logical slot, so looped weight sharing still lets
        # each depth learn its own stream routing.
        self.hyper_connections = (
            nn.ModuleList(
                [_HyperConnections(int(mhc_expansion)) for _ in range(logical_layer_count)]
            )
            if int(mhc_expansion) > 1
            else None
        )
        horizons = sorted({int(h) for h in (mtp_horizons or [1])})
        self.mtp_heads = nn.ModuleDict(
            {str(h): nn.Linear(dim, vocab_size) for h in horizons if h > 1}
        )
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)

    def _run_layer(
        self, layer: nn.Module, hidden: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """One block application, optionally under activation checkpointing."""
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                lambda tensor: layer(tensor, src_mask=mask), hidden, use_reentrant=False
            )
        return layer(hidden, src_mask=mask)

    def _forward_hidden(self, inputs: torch.Tensor) -> torch.Tensor:
        """Embed, run every logical layer, and return the final normed hidden state."""
        sequence_length = inputs.size(1)
        if sequence_length > self._causal_mask.size(0):
            raise ValueError(
                f"Sequence length {sequence_length} exceeds configured limit "
                f"{self._causal_mask.size(0)}"
            )
        positions = torch.arange(sequence_length, device=inputs.device)
        hidden = self.embedding(inputs) + self.position_embedding(positions).unsqueeze(0)
        if self.engram is not None:
            hidden = hidden + self.engram(inputs)
        if self.embedding_scale:
            hidden = hidden * math.sqrt(hidden.size(-1))
        causal_mask = self._causal_mask[:sequence_length, :sequence_length].to(inputs.device)
        if self.hyper_connections is not None:
            streams = hidden.unsqueeze(0).expand(
                self.hyper_connections[0].rate, -1, -1, -1
            )
            for index, layer in enumerate(self.logical_layers):
                mixer = self.hyper_connections[index]
                block_input = mixer.read_input(streams)
                block_output = self._run_layer(layer, block_input, causal_mask)
                streams = mixer.write_output(streams, block_output - block_input)
            hidden = self.hyper_connections[-1].read_input(streams)
        else:
            residual_anchor = hidden
            for index, layer in enumerate(self.logical_layers):
                hidden = self._run_layer(layer, hidden, causal_mask)
                if self.attn_res_block_size > 0 and (index + 1) % self.attn_res_block_size == 0:
                    hidden = hidden + residual_anchor
                    residual_anchor = hidden
        return self.norm(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc_out(self._forward_hidden(inputs))

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy plus weighted auxiliary multi-token-prediction losses.

        ``targets`` follows the executor convention ``targets[t] == inputs[t+1]``,
        so the label for horizon ``k`` at position ``t`` is ``targets[t + k - 1]``.
        """
        hidden = self._forward_hidden(inputs)
        logits = self.fc_out(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        aux_losses = []
        for name, head in self.mtp_heads.items():
            shift = int(name) - 1
            if shift <= 0 or shift >= targets.size(1):
                continue
            horizon_logits = head(hidden[:, :-shift])
            horizon_labels = targets[:, shift:]
            aux_losses.append(
                torch.nn.functional.cross_entropy(
                    horizon_logits.reshape(-1, horizon_logits.size(-1)),
                    horizon_labels.reshape(-1),
                )
            )
        if aux_losses:
            loss = loss + MTP_AUX_LOSS_WEIGHT * torch.stack(aux_losses).mean()
        return loss

class GatedGRULanguageModel(nn.Module):
    """GRU recurrent stack aimed at low-memory / long-context workloads."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.0,
        seq_len: int = 2048,
        embedding_scale: bool = False,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding_scale = bool(embedding_scale)
        self.recurrent = nn.GRU(
            dim,
            dim,
            num_layers=max(1, gru_layers),
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) > self.seq_len:
            raise ValueError(f"Sequence length {inputs.size(1)} exceeds configured limit {self.seq_len}")
        hidden = self.embedding(inputs)
        if self.embedding_scale:
            hidden = hidden * math.sqrt(hidden.size(-1))
        hidden, _ = self.recurrent(hidden)
        return self.fc_out(self.norm(hidden))


class GatedConvLanguageModel(nn.Module):
    """Left-padded gated convolutions with residual mixing for LM tasks."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        n_logical_layers: int = 6,
        conv_kernel: int = 5,
        dropout: float = 0.0,
        seq_len: int = 2048,
        embedding_scale: bool = False,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding_scale = bool(embedding_scale)
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(dim, dim * 2, kernel_size=conv_kernel)
                for _ in range(max(1, n_logical_layers))
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)
        self.conv_kernel = conv_kernel
        self.seq_len = seq_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) > self.seq_len:
            raise ValueError(f"Sequence length {inputs.size(1)} exceeds configured limit {self.seq_len}")
        hidden = self.embedding(inputs)
        if self.embedding_scale:
            hidden = hidden * math.sqrt(hidden.size(-1))
        hidden = hidden.transpose(1, 2)
        for block in self.blocks:
            # Left-pad so each conv stays causal; output length matches the input.
            padded = torch.nn.functional.pad(hidden, (self.conv_kernel - 1, 0))
            gated = block(padded).transpose(1, 2)
            values, gate = gated.chunk(2, dim=-1)
            hidden = (hidden.transpose(1, 2) + self.dropout(values * torch.sigmoid(gate))).transpose(1, 2)
        return self.fc_out(self.norm(hidden.transpose(1, 2)))


class _MoEBlock(nn.Module):
    """Causal self-attention followed by a sparsely routed expert FFN."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        dropout: float,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        use_head_gating: bool = False,
    ) -> None:
        super().__init__()
        self.attn = _CausalSelfAttention(
            dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            use_head_gating=use_head_gating,
        )
        self.norm = nn.LayerNorm(dim)
        self.router = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.top_k = top_k

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.attn(hidden, attn_mask)
        normalized = self.norm(hidden)
        scores = torch.softmax(self.router(normalized), dim=-1)
        values, indices = scores.topk(min(self.top_k, scores.size(-1)), dim=-1)
        values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mixed = torch.zeros_like(hidden)
        # Collapse batch×seq so routing masks index tokens, not the top-k axis.
        flat_normalized = normalized.reshape(-1, normalized.size(-1))
        flat_indices = indices.reshape(-1, indices.size(-1))
        flat_values = values.reshape(-1, values.size(-1))
        flat_mixed = mixed.reshape(-1, mixed.size(-1))
        for expert_index, expert in enumerate(self.experts):
            selected = (flat_indices == expert_index).any(dim=-1)
            if selected.any():
                expert_output = expert(flat_normalized[selected])
                expert_weights = torch.where(
                    flat_indices[selected] == expert_index,
                    flat_values[selected],
                    torch.zeros_like(flat_values[selected]),
                ).sum(dim=-1, keepdim=True)
                flat_mixed[selected] += expert_output * expert_weights
        return hidden + flat_mixed.reshape_as(hidden)


class MoEDecoderLanguageModel(nn.Module):
    """Sparse MoE decoder: causal attention plus top-k expert routing per block."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 256,
        ffn_dim: int = 1024,
        n_logical_layers: int = 8,
        num_experts: int = 4,
        top_k: int = 2,
        seq_len: int = 2048,
        dropout: float = 0.0,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        use_head_gating: bool = False,
        embedding_scale: bool = False,
        sliding_window: int = 0,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(seq_len, dim)
        self.embedding_scale = bool(embedding_scale)
        self.register_buffer(
            "_causal_mask",
            build_causal_mask(seq_len, max(0, int(sliding_window))),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [
                _MoEBlock(
                    dim,
                    ffn_dim,
                    num_experts,
                    top_k,
                    dropout,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    qk_norm=qk_norm,
                    use_head_gating=use_head_gating,
                )
                for _ in range(max(1, n_logical_layers))
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.size(1)
        if sequence_length > self.seq_len:
            raise ValueError(f"Sequence length {sequence_length} exceeds configured limit {self.seq_len}")
        positions = torch.arange(sequence_length, device=inputs.device)
        hidden = self.embedding(inputs) + self.position_embedding(positions).unsqueeze(0)
        if self.embedding_scale:
            hidden = hidden * math.sqrt(hidden.size(-1))
        causal_mask = self._causal_mask[:sequence_length, :sequence_length].to(inputs.device)
        for block in self.blocks:
            hidden = block(hidden, causal_mask)
        return self.fc_out(self.norm(hidden))


class LSTMLanguageModel(nn.Module):
    """Stacked LSTM language model (API sibling of the GRU family)."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.0,
        seq_len: int = 2048,
        embedding_scale: bool = False,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        layers = max(1, gru_layers)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding_scale = bool(embedding_scale)
        self.recurrent = nn.LSTM(
            dim,
            dim,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) > self.seq_len:
            raise ValueError(f"Sequence length {inputs.size(1)} exceeds configured limit {self.seq_len}")
        hidden = self.embedding(inputs)
        if self.embedding_scale:
            hidden = hidden * math.sqrt(hidden.size(-1))
        hidden, _ = self.recurrent(hidden)
        return self.fc_out(self.norm(hidden))


class _ResidualConvBlock(nn.Module):
    """Causal residual 1D conv block without gating."""

    def __init__(self, dim: int, conv_kernel: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=conv_kernel)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.conv_kernel = conv_kernel

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # Expect layout (batch, channels, sequence).
        residual = hidden
        padded = torch.nn.functional.pad(hidden, (self.conv_kernel - 1, 0))
        activated = torch.relu(self.conv(padded))
        projected = self.proj(activated)
        mixed = residual + self.dropout(projected)
        return self.norm(mixed.transpose(1, 2)).transpose(1, 2)


class ResNetLMLanguageModel(nn.Module):
    """Stack of residual causal 1D convolutions for next-token modeling."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        n_logical_layers: int = 6,
        conv_kernel: int = 5,
        dropout: float = 0.0,
        seq_len: int = 2048,
        embedding_scale: bool = False,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.embedding_scale = bool(embedding_scale)
        self.blocks = nn.ModuleList(
            [
                _ResidualConvBlock(dim, conv_kernel, dropout)
                for _ in range(max(1, n_logical_layers))
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.size(1) > self.seq_len:
            raise ValueError(f"Sequence length {inputs.size(1)} exceeds configured limit {self.seq_len}")
        hidden = self.embedding(inputs)
        if self.embedding_scale:
            hidden = hidden * math.sqrt(hidden.size(-1))
        hidden = hidden.transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        return self.fc_out(self.norm(hidden.transpose(1, 2)))


class _HybridAttnBlock(nn.Module):
    """Attention half of the hybrid family: GQA-capable attention plus FFN."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        use_head_gating: bool = False,
    ) -> None:
        super().__init__()
        self.attn = _CausalSelfAttention(
            dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            use_head_gating=use_head_gating,
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.attn(hidden, attn_mask)
        return hidden + self.ffn(self.norm2(hidden))


class _HybridGRUBlock(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gru = nn.GRU(dim, dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        recurrent, _ = self.gru(self.norm(hidden))
        return residual + self.dropout(recurrent)


class HybridAttnRNNLanguageModel(nn.Module):
    """Interleaved causal attention blocks and residual GRU blocks.

    The attention halves honor the same GQA / QK-norm / head-gating /
    sliding-window knobs as the decoder family, matching what the
    orchestrator validates for ``hybrid_attn_rnn`` configs.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        n_logical_layers: int = 6,
        n_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.0,
        seq_len: int = 2048,
        n_kv_heads: int | None = None,
        qk_norm: bool = False,
        use_head_gating: bool = False,
        embedding_scale: bool = False,
        sliding_window: int = 0,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        heads = n_heads if dim % max(1, n_heads) == 0 else 1
        self.embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(seq_len, dim)
        self.embedding_scale = bool(embedding_scale)
        self.register_buffer(
            "_causal_mask",
            build_causal_mask(seq_len, max(0, int(sliding_window))),
            persistent=False,
        )
        blocks: list[nn.Module] = []
        for index in range(max(1, n_logical_layers)):
            if index % 2 == 0:
                blocks.append(
                    _HybridAttnBlock(
                        dim,
                        heads,
                        ffn_dim,
                        dropout,
                        n_kv_heads=n_kv_heads,
                        qk_norm=qk_norm,
                        use_head_gating=use_head_gating,
                    )
                )
            else:
                blocks.append(_HybridGRUBlock(dim, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.size(1)
        if sequence_length > self.seq_len:
            raise ValueError(f"Sequence length {sequence_length} exceeds configured limit {self.seq_len}")
        positions = torch.arange(sequence_length, device=inputs.device)
        hidden = self.embedding(inputs) + self.position_embedding(positions).unsqueeze(0)
        if self.embedding_scale:
            hidden = hidden * math.sqrt(hidden.size(-1))
        causal_mask = self._causal_mask[:sequence_length, :sequence_length].to(inputs.device)
        for block in self.blocks:
            if isinstance(block, _HybridAttnBlock):
                hidden = block(hidden, causal_mask)
            else:
                hidden = block(hidden)
        return self.fc_out(self.norm(hidden))


class _DenseFFNBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.ffn(self.norm(hidden))


class DenseFFNLanguageModel(nn.Module):
    """Deep residual feed-forward tower on embeddings (attention-free)."""

    def __init__(
        self,
        vocab_size: int = 256,
        dim: int = 128,
        ffn_dim: int = 512,
        n_logical_layers: int = 8,
        dropout: float = 0.0,
        seq_len: int = 2048,
        embedding_scale: bool = False,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(seq_len, dim)
        self.embedding_scale = bool(embedding_scale)
        self.blocks = nn.ModuleList(
            [_DenseFFNBlock(dim, ffn_dim, dropout) for _ in range(max(1, n_logical_layers))]
        )
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.size(1)
        if sequence_length > self.seq_len:
            raise ValueError(f"Sequence length {sequence_length} exceeds configured limit {self.seq_len}")
        positions = torch.arange(sequence_length, device=inputs.device)
        hidden = self.embedding(inputs) + self.position_embedding(positions).unsqueeze(0)
        if self.embedding_scale:
            hidden = hidden * math.sqrt(hidden.size(-1))
        for block in self.blocks:
            hidden = block(hidden)
        return self.fc_out(self.norm(hidden))


_ARCHITECTURE_FAMILIES = {
    "decoder_transformer": DistribAITinyLanguageModel,
    "gru": GatedGRULanguageModel,
    "gated_conv": GatedConvLanguageModel,
    "moe_decoder": MoEDecoderLanguageModel,
    "lstm": LSTMLanguageModel,
    "resnet_lm": ResNetLMLanguageModel,
    "hybrid_attn_rnn": HybridAttnRNNLanguageModel,
    "dense_ffn": DenseFFNLanguageModel,
}


class DistribAIModelWrapper(nn.Module):
    """Facade that builds a native DistribAI LM from a profile or custom config."""

    MODEL_CONFIGS: dict[str, dict[str, int | str]] = {
        "distribai-tiny": {"architecture": "decoder_transformer", "dim": 64, "n_unique_layers": 4, "n_logical_layers": 8, "n_heads": 4, "ffn_dim": 256},
        "distribai-small": {"architecture": "decoder_transformer", "dim": 128, "n_unique_layers": 6, "n_logical_layers": 12, "n_heads": 4, "ffn_dim": 512},
        "distribai-base": {"architecture": "decoder_transformer", "dim": 256, "n_unique_layers": 8, "n_logical_layers": 16, "n_heads": 8, "ffn_dim": 1024},
        "distribai-medium": {"architecture": "decoder_transformer", "dim": 384, "n_unique_layers": 10, "n_logical_layers": 20, "n_heads": 8, "ffn_dim": 1536},
        "distribai-large": {"architecture": "decoder_transformer", "dim": 768, "n_unique_layers": 16, "n_logical_layers": 32, "n_heads": 12, "ffn_dim": 3072},
        "distribai-xl": {"architecture": "decoder_transformer", "dim": 1024, "n_unique_layers": 24, "n_logical_layers": 48, "n_heads": 16, "ffn_dim": 4096},
        "distribai-lstm-small": {"architecture": "lstm", "dim": 128, "gru_layers": 2, "ffn_dim": 512},
        "distribai-gru-small": {"architecture": "gru", "dim": 128, "gru_layers": 2, "ffn_dim": 512},
        "distribai-conv-small": {
            "architecture": "gated_conv",
            "dim": 128,
            "n_logical_layers": 6,
            "conv_kernel": 5,
            "ffn_dim": 512,
        },
        "distribai-moe-small": {
            "architecture": "moe_decoder",
            "dim": 128,
            "ffn_dim": 512,
            "n_logical_layers": 4,
            "n_heads": 4,
            "num_experts": 4,
            "top_k": 2,
        },
        "distribai-resnet-tiny": {
            "architecture": "resnet_lm",
            "dim": 64,
            "n_logical_layers": 4,
            "conv_kernel": 5,
            "ffn_dim": 256,
        },
        "distribai-hybrid-small": {
            "architecture": "hybrid_attn_rnn",
            "dim": 128,
            "n_logical_layers": 6,
            "n_heads": 4,
            "ffn_dim": 512,
        },
        "distribai-dense-tiny": {
            "architecture": "dense_ffn",
            "dim": 64,
            "n_logical_layers": 6,
            "ffn_dim": 256,
        },
    }

    @classmethod
    def register_model_config(cls, name: str, config: dict[str, Any]) -> None:
        cls.MODEL_CONFIGS[name.lower()] = config
        logger.info("Registered DistribAI model architecture: %s", name)

    def __init__(
        self,
        model_name: str,
        vocab_size: int = 256,
        seq_len: int | None = None,
        custom_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_name = model_name.lower()
        if custom_config is not None:
            config_values = dict(custom_config)
        elif self.model_name in self.MODEL_CONFIGS:
            config_values = dict(self.MODEL_CONFIGS[self.model_name])
        else:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(self.MODEL_CONFIGS)}")
        config_values.setdefault("family", config_values.get("architecture", "decoder_transformer"))
        # Uploaded architectures control their own context window. Named profiles
        # still default to 2048 tokens when seq_len is omitted.
        config_values.setdefault("seq_len", seq_len if seq_len is not None else 2048)
        config_values.update(kwargs)
        config_values = validate_architecture_config(config_values)
        # When callers omit n_kv_heads, match n_heads so GQA stays opt-in.
        if "n_kv_heads" not in config_values:
            config_values["n_kv_heads"] = int(config_values.get("n_heads", 4))
        self.config = ModelConfig(**{key: value for key, value in config_values.items() if key in ModelConfig.__dataclass_fields__})
        family_cls = _ARCHITECTURE_FAMILIES[self.config.family]
        self.model = family_cls(
            vocab_size=vocab_size,
            dim=self.config.dim,
            n_unique_layers=self.config.n_unique_layers,
            n_logical_layers=self.config.n_logical_layers,
            n_heads=self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads,
            ffn_dim=self.config.ffn_dim,
            dropout=self.config.dropout,
            seq_len=self.config.seq_len,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            conv_kernel=self.config.conv_kernel,
            gru_layers=self.config.gru_layers,
            qk_norm=self.config.qk_norm,
            use_head_gating=self.config.use_head_gating,
            embedding_scale=self.config.embedding_scale,
            attn_res_block_size=self.config.attn_res_block_size,
            sliding_window=self.config.sliding_window,
            engram_dim=self.config.engram_dim,
            mhc_expansion=self.config.mhc_expansion,
            mtp_horizons=list(self.config.mtp_horizons),
            grad_checkpoint=self.config.grad_checkpoint,
        )
        logger.info("Created DistribAI %s model: %s params", model_name, f"{self.param_count():,}")

    def param_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def unique_layers(self) -> nn.ModuleList:
        """Return the family's ``layers`` ModuleList when present, else empty."""
        return getattr(self.model, "layers", nn.ModuleList())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.model(inputs)
        return output[0] if isinstance(output, tuple) else output

    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Family-aware training loss.

        Decoder models with multi-token-prediction heads add their auxiliary
        losses; every other family reduces to plain next-token cross-entropy.
        """
        model_loss = getattr(self.model, "compute_loss", None)
        if callable(model_loss):
            return model_loss(inputs, targets)
        logits = self.forward(inputs)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )

    def get_gradient_norm(self) -> float:
        total_norm = 0.0
        for parameter in self.parameters():
            if parameter.grad is not None:
                total_norm += parameter.grad.norm().item() ** 2
        return total_norm**0.5

    def clip_gradients(self, max_norm: float = 1.0) -> None:
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

    def save_checkpoint(self, path: str, step: int, loss: float) -> None:
        torch.save({"model_state": self.state_dict(), "config": self.config, "step": step, "loss": loss}, path)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["model_state"])
        return checkpoint


class CustomModelBuilder:
    """Helpers that assemble legacy decoder-transformer sizes via the wrapper."""

    @staticmethod
    def create_custom_model(
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        ffn_dim: int | None = None,
        vocab_size: int = 256,
        seq_len: int = 2048,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> DistribAIModelWrapper:
        config = {
            "family": "decoder_transformer",
            "dim": dim,
            "n_unique_layers": n_layers,
            "n_logical_layers": n_layers * 2,
            "n_heads": n_heads,
            "n_kv_heads": int(kwargs.pop("n_kv_heads", n_heads)),
            "ffn_dim": ffn_dim or 4 * dim,
            "dropout": dropout,
            "seq_len": seq_len,
            **kwargs,
        }
        return DistribAIModelWrapper("custom", vocab_size=vocab_size, seq_len=seq_len, custom_config=config)

    @staticmethod
    def create_tiny_model(vocab_size: int = 256) -> DistribAIModelWrapper:
        return CustomModelBuilder.create_custom_model(dim=64, n_layers=4, n_heads=4, ffn_dim=256, vocab_size=vocab_size, seq_len=512)

    @staticmethod
    def create_small_model(vocab_size: int = 256) -> DistribAIModelWrapper:
        return CustomModelBuilder.create_custom_model(dim=256, n_layers=8, n_heads=8, ffn_dim=1024, vocab_size=vocab_size, seq_len=1024)

    @staticmethod
    def create_medium_model(vocab_size: int = 256) -> DistribAIModelWrapper:
        return CustomModelBuilder.create_custom_model(dim=768, n_layers=16, n_heads=12, ffn_dim=3072, vocab_size=vocab_size, seq_len=2048)


def get_model(
    model_name: str,
    vocab_size: int = 256,
    architecture_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate a native model from a legacy profile name or architecture dict."""
    normalized = model_name.strip().lower()
    if architecture_config is not None:
        return DistribAIModelWrapper(
            normalized or "uploaded-architecture",
            vocab_size=vocab_size,
            custom_config=architecture_config,
            **kwargs,
        )
    if normalized in DistribAIModelWrapper.MODEL_CONFIGS:
        return DistribAIModelWrapper(normalized, vocab_size=vocab_size, **kwargs)
    if normalized == "custom":
        return CustomModelBuilder.create_custom_model(vocab_size=vocab_size, **kwargs)
    if normalized == "tiny":
        return CustomModelBuilder.create_tiny_model(vocab_size)
    if normalized == "small":
        return CustomModelBuilder.create_small_model(vocab_size)
    if normalized == "medium":
        return CustomModelBuilder.create_medium_model(vocab_size)
    raise ValueError(f"Unknown model: {model_name}")
