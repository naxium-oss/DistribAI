"""Portable weight export helpers (safetensors / ONNX) for native DistribAI models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_state_dict_torch(model: nn.Module, path: str | Path) -> Path:
    """Write a standard PyTorch ``state_dict`` checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, target)
    return target


def export_safetensors(model: nn.Module, path: str | Path) -> Path:
    """Export weights as safetensors when the optional dependency is installed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is not installed; pip install safetensors to enable this export"
        ) from exc
    tensors = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(tensors, str(target))
    return target


def export_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    seq_len: int = 8,
    vocab_size: int = 256,
    opset: int = 17,
) -> Path:
    """Export a causal LM-style module that accepts LongTensor token ids."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()
    dummy = torch.randint(0, max(2, vocab_size), (1, seq_len), dtype=torch.long)
    torch.onnx.export(
        model,
        dummy,
        str(target),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=opset,
    )
    return target


def export_portable_weights(
    model: nn.Module,
    directory: str | Path,
    *,
    basename: str = "model",
    formats: tuple[str, ...] = ("torch", "safetensors", "onnx"),
    seq_len: int = 8,
    vocab_size: int = 256,
) -> dict[str, Any]:
    """Best-effort multi-format export; skips optional formats that are unavailable."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for fmt in formats:
        try:
            if fmt == "torch":
                results[fmt] = str(export_state_dict_torch(model, root / f"{basename}.pt"))
            elif fmt == "safetensors":
                results[fmt] = str(export_safetensors(model, root / f"{basename}.safetensors"))
            elif fmt == "onnx":
                results[fmt] = str(
                    export_onnx(
                        model,
                        root / f"{basename}.onnx",
                        seq_len=seq_len,
                        vocab_size=vocab_size,
                    )
                )
            else:
                results[fmt] = {"error": f"unknown format: {fmt}"}
        except Exception as exc:
            logger.info("export format %s skipped: %s", fmt, exc)
            results[fmt] = {"error": str(exc)}
    return results
