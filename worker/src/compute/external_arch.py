"""Load arbitrary external model architectures (Hugging Face custom code, local dirs).

This path is for models that are not one of DistribAI's native declarative families.
Any Hub repo or local folder with a transformers-compatible config (including
``auto_map`` / remote code) can be loaded when explicitly allowed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def external_arch_allowed(explicit: bool | None = None) -> bool:
    """Return whether external/custom-code architectures may be loaded.

    Prefer an explicit job flag when provided; otherwise honor
    ``DISTRIBAI_ALLOW_EXTERNAL_ARCH`` (default off).
    """
    if explicit is not None:
        return bool(explicit)
    return os.getenv("DISTRIBAI_ALLOW_EXTERNAL_ARCH", "").lower() in {"1", "true", "yes"}


def looks_like_external_model_ref(value: str | None) -> bool:
    """Heuristic: Hub ids (``org/name``) or existing local paths."""
    if not value or not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text.lower() in {"custom", "tiny", "small", "medium", "toy"}:
        return False
    if text.startswith(("http://", "https://", "hf://")):
        return True
    if os.path.isdir(text) or os.path.isfile(text):
        return True
    # Hub-style org/repo (reject bare profile names without a slash)
    if "/" in text and not text.startswith("/") and " " not in text:
        return True
    return False


def normalize_model_ref(value: str) -> str:
    """Strip optional ``hf://`` prefix from a model reference."""
    text = value.strip()
    if text.startswith("hf://"):
        return text[5:]
    return text


def _resolve_torch_dtype(torch_dtype: str | None) -> Any:
    if not torch_dtype:
        return None
    import torch

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(str(torch_dtype).lower())


def load_external_architecture(
    source: str,
    *,
    trust_remote_code: bool = True,
    cache_dir: str | None = None,
    torch_dtype: str | None = None,
    allow: bool | None = None,
    config_overrides: dict[str, Any] | None = None,
    from_scratch: bool = False,
) -> Any:
    """Instantiate an arbitrary transformers causal LM (or base model fallback).

    ``from_scratch`` builds a randomly-initialized model from the resolved
    architecture (optionally after applying ``config_overrides``) instead of
    downloading pretrained weights — useful for training/fine-tuning a job's
    own data on an architecture whose published checkpoint is large, gated,
    or simply not the point (DistribAI jobs bring their own weights/data).

    Raises:
        PermissionError: When external architectures are not allowed.
        RuntimeError: When transformers is missing.
        ValueError: When ``source`` is empty.
    """
    if not external_arch_allowed(allow):
        raise PermissionError(
            "External architectures are disabled. Set DISTRIBAI_ALLOW_EXTERNAL_ARCH=1 "
            "or pass allow_external_arch=true on the job."
        )
    ref = normalize_model_ref(source)
    if not ref:
        raise ValueError("external architecture source is empty")

    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required to load external architectures") from exc

    trust = bool(trust_remote_code)
    dtype = _resolve_torch_dtype(torch_dtype)
    kwargs: dict[str, Any] = {"trust_remote_code": trust, "cache_dir": cache_dir}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    logger.info(
        "Loading external architecture from %s (trust_remote_code=%s, from_scratch=%s)",
        ref,
        trust,
        from_scratch,
    )
    # Resolve config first so custom auto_map classes are registered before weight load.
    config = AutoConfig.from_pretrained(ref, trust_remote_code=trust, cache_dir=cache_dir)
    if config_overrides:
        for key, value in config_overrides.items():
            setattr(config, key, value)

    if from_scratch:
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        try:
            return AutoModelForCausalLM.from_config(config, **model_kwargs)
        except (ValueError, AttributeError, TypeError) as causal_exc:
            logger.warning(
                "AutoModelForCausalLM.from_config failed for %s (%s); falling back to AutoModel",
                ref,
                causal_exc,
            )
            return AutoModel.from_config(config, **model_kwargs)

    try:
        return AutoModelForCausalLM.from_pretrained(ref, config=config, **kwargs)
    except (ValueError, OSError, AttributeError) as causal_exc:
        logger.warning(
            "AutoModelForCausalLM failed for %s (%s); falling back to AutoModel",
            ref,
            causal_exc,
        )
        return AutoModel.from_pretrained(ref, config=config, **kwargs)
