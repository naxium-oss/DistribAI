"""Bounds-check and normalize declarative model architecture payloads.

Configs are data-only: they pick a registered worker family and supply capped
numeric knobs. They must not carry import paths, Python class names, or any
executable material.
"""

from __future__ import annotations

import json
import math
from typing import Any

ARCHITECTURE_CONFIG_VERSION = 1
SUPPORTED_ARCHITECTURE_FAMILIES = frozenset(
    {
        "decoder_transformer",
        "gru",
        "gated_conv",
        "moe_decoder",
        "lstm",
        "resnet_lm",
        "hybrid_attn_rnn",
        "dense_ffn",
    }
)
MAX_ARCHITECTURE_CONFIG_BYTES = 64 * 1024
MAX_ARCHITECTURE_CONFIG_DEPTH = 6
MAX_ESTIMATED_PARAMETERS = 512_000_000
MAX_TRANSFORMER_SEQ_LEN = 8192

_INT_LIMITS = {
    "dim": (16, 4096),
    "n_unique_layers": (1, 64),
    "n_logical_layers": (1, 128),
    "n_heads": (1, 64),
    "n_kv_heads": (1, 64),
    "ffn_dim": (16, 16384),
    "seq_len": (8, 32768),
    "sliding_window": (0, 32768),
    "engram_dim": (0, 1024),
    "mhc_expansion": (1, 16),
    "num_experts": (1, 64),
    "top_k": (1, 16),
    "conv_kernel": (2, 31),
    "gru_layers": (1, 16),
}
_FLOAT_LIMITS = {"dropout": (0.0, 0.5)}
_ALLOWED_KEYS = {"version", "family", "architecture", *_INT_LIMITS, *_FLOAT_LIMITS}


def _container_nesting(value: Any) -> int:
    """Walk dict/list nesting with an explicit stack (no recursive walk of untrusted input)."""
    deepest = 0
    pending = [(value, 0)]
    while pending:
        node, level = pending.pop()
        if not isinstance(node, (dict, list)):
            continue
        deepest = max(deepest, level)
        kids = node.values() if isinstance(node, dict) else node
        pending.extend((kid, level + 1) for kid in kids)
    return deepest


def _rough_parameter_count(config: dict[str, Any]) -> int:
    """Cheap native-size estimate used to reject configs before workers allocate."""
    dim = int(config.get("dim", 256))
    ffn_dim = int(config.get("ffn_dim", 4 * dim))
    layers = int(config.get("n_unique_layers", config.get("n_logical_layers", 8)))
    vocab = 256
    family = config["family"]
    if family == "decoder_transformer":
        return layers * (4 * dim * dim + 2 * dim * ffn_dim) + 2 * vocab * dim
    if family == "gru":
        gru_layers = int(config.get("gru_layers", 2))
        return gru_layers * 3 * (dim * dim + dim * dim + 2 * dim) + 2 * vocab * dim
    if family == "lstm":
        # Four gates per layer; depth still keyed by gru_layers for API compatibility.
        lstm_layers = int(config.get("gru_layers", 2))
        return lstm_layers * 4 * (dim * dim + dim * dim + 2 * dim) + 2 * vocab * dim
    if family == "gated_conv":
        kernel = int(config.get("conv_kernel", 5))
        conv_layers = int(config.get("n_logical_layers", 6))
        return conv_layers * 2 * dim * dim * kernel + 2 * vocab * dim
    if family == "resnet_lm":
        kernel = int(config.get("conv_kernel", 5))
        res_layers = int(config.get("n_logical_layers", 6))
        # Per block: residual depthwise-style 1D conv plus a pointwise map (single width).
        return res_layers * (dim * dim * kernel + dim * dim) + 2 * vocab * dim
    if family == "hybrid_attn_rnn":
        hybrid_layers = int(config.get("n_logical_layers", 8))
        attn_layers = (hybrid_layers + 1) // 2
        rnn_layers = hybrid_layers // 2
        return (
            attn_layers * (4 * dim * dim + 2 * dim * ffn_dim)
            + rnn_layers * 3 * (dim * dim + dim * dim + 2 * dim)
            + 2 * vocab * dim
        )
    if family == "dense_ffn":
        ffn_layers = int(config.get("n_logical_layers", 8))
        return ffn_layers * (2 * dim * ffn_dim) + 2 * vocab * dim
    experts = int(config.get("num_experts", 4))
    moe_layers = int(config.get("n_logical_layers", 8))
    return moe_layers * (experts * 2 * dim * ffn_dim + dim * experts) + 2 * vocab * dim


def validate_architecture_config(value: Any) -> dict[str, Any]:
    """Normalize a declarative architecture object after applying safety bounds.

    Raises:
        ValueError: When the payload is oversized, unsupported, or inconsistent.
    """
    if not isinstance(value, dict):
        raise ValueError("architecture_config must be an object")
    if _container_nesting(value) > MAX_ARCHITECTURE_CONFIG_DEPTH:
        raise ValueError("architecture_config is nested too deeply")
    try:
        encoded_size = len(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError("architecture_config must contain JSON-finite values") from exc
    if encoded_size > MAX_ARCHITECTURE_CONFIG_BYTES:
        raise ValueError("architecture_config exceeds 64 KiB")
    unknown = set(value) - _ALLOWED_KEYS
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
    raw_version = normalized.get("version", ARCHITECTURE_CONFIG_VERSION)
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ValueError("architecture_config.version must be an integer")
    normalized["version"] = raw_version
    if normalized["version"] != ARCHITECTURE_CONFIG_VERSION:
        raise ValueError(f"unsupported architecture_config version: {normalized['version']}")
    family = normalized.get("family", normalized.get("architecture"))
    if not isinstance(family, str) or family.strip().lower() not in SUPPORTED_ARCHITECTURE_FAMILIES:
        raise ValueError(
            "architecture_config.family must be one of: "
            + ", ".join(sorted(SUPPORTED_ARCHITECTURE_FAMILIES))
        )
    normalized["family"] = family.strip().lower()
    normalized["architecture"] = normalized["family"]

    for key, (minimum, maximum) in _INT_LIMITS.items():
        if key not in normalized:
            continue
        value_for_key = normalized[key]
        if isinstance(value_for_key, bool) or not isinstance(value_for_key, int):
            raise ValueError(f"architecture_config.{key} must be an integer")
        if not minimum <= value_for_key <= maximum:
            raise ValueError(f"architecture_config.{key} must be between {minimum} and {maximum}")
    for key, (minimum, maximum) in _FLOAT_LIMITS.items():
        if key not in normalized:
            continue
        value_for_key = normalized[key]
        if isinstance(value_for_key, bool) or not isinstance(value_for_key, (int, float)):
            raise ValueError(f"architecture_config.{key} must be numeric")
        if not math.isfinite(float(value_for_key)) or not minimum <= float(value_for_key) <= maximum:
            raise ValueError(f"architecture_config.{key} must be between {minimum} and {maximum}")
        normalized[key] = float(value_for_key)

    if normalized["family"] in {"decoder_transformer", "hybrid_attn_rnn"}:
        dim = int(normalized.get("dim", 256))
        heads = int(normalized.get("n_heads", 8))
        if dim % heads:
            raise ValueError("architecture_config.dim must be divisible by n_heads")
        if int(normalized.get("seq_len", 512)) > MAX_TRANSFORMER_SEQ_LEN:
            raise ValueError(
                f"architecture_config.seq_len must be at most {MAX_TRANSFORMER_SEQ_LEN} for transformer attention"
            )
    if normalized["family"] == "decoder_transformer" and int(
        normalized.get("n_logical_layers", normalized.get("n_unique_layers", 8))
    ) < int(normalized.get("n_unique_layers", 8)):
        raise ValueError("n_logical_layers cannot be less than n_unique_layers")
    if normalized["family"] == "moe_decoder":
        experts = int(normalized.get("num_experts", 4))
        top_k = int(normalized.get("top_k", 2))
        if top_k > experts:
            raise ValueError("architecture_config.top_k cannot exceed num_experts")
    estimated_parameters = _rough_parameter_count(normalized)
    if estimated_parameters > MAX_ESTIMATED_PARAMETERS:
        raise ValueError(
            "architecture_config estimated parameter count exceeds "
            f"{MAX_ESTIMATED_PARAMETERS:,}"
        )
    return normalized


def architecture_config_from_json(text: str) -> dict[str, Any]:
    """Decode a UTF-8 JSON architecture document and run full validation."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_ARCHITECTURE_CONFIG_BYTES:
        raise ValueError("architecture config file exceeds 64 KiB")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid architecture JSON: {exc.msg}") from exc
    return validate_architecture_config(value)
