"""Portable weight export helpers (safetensors / ONNX) for native DistribAI models.

Optional dependencies are loaded lazily:

* ``safetensors`` — required for ``export_safetensors`` (``pip install safetensors``)
* ONNX export uses ``torch.onnx`` (core torch); the ``onnx`` package is optional
  and only used when verifying the written graph.

CLI::

    python -m worker.src.compute.export_weights \\
        --model distribai-tiny --format safetensors --out weights.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class ExportDependencyError(ImportError):
    """Raised when an optional export backend is not installed."""


def _state_dict_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}


def export_safetensors(model: nn.Module, path: str | Path) -> Path:
    """Write model weights to a ``.safetensors`` file.

    Raises ``ExportDependencyError`` when the ``safetensors`` package is absent.
    """
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ExportDependencyError(
            "safetensors is not installed; run: pip install safetensors"
        ) from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(_state_dict_tensors(model), str(out))
    return out


def export_onnx(
    model: nn.Module,
    path: str | Path,
    *,
    batch_size: int = 1,
    seq_len: int = 8,
    vocab_hint: int | None = None,
    opset: int = 17,
) -> Path:
    """Export a DistribAI language model to ONNX via ``torch.onnx.export``.

    Requires the optional ``onnx`` package (``pip install onnx`` or
    ``pip install 'distribai[export]'``). Builds a dummy ``LongTensor`` input of
    shape ``(batch_size, seq_len)``. Prefer families that trace cleanly
    (``dense_ffn`` / ``gru``) for smoke exports.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ExportDependencyError(
            "onnx is not installed; run: pip install onnx"
        ) from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()

    if vocab_hint is None:
        embedding = getattr(getattr(model, "model", model), "embedding", None)
        if embedding is not None and hasattr(embedding, "num_embeddings"):
            vocab_hint = int(embedding.num_embeddings)
        else:
            vocab_hint = 256

    dummy = torch.randint(0, max(1, vocab_hint), (batch_size, seq_len), dtype=torch.long)

    export_kwargs: dict[str, Any] = {
        "input_names": ["input_ids"],
        "output_names": ["logits"],
        "opset_version": opset,
        "do_constant_folding": True,
    }
    try:
        try:
            torch.onnx.export(model, dummy, str(out), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(out), **export_kwargs)
    except Exception as exc:
        # Torch wraps a missing onnx install as OnnxExporterError on some versions.
        message = str(exc).lower()
        if "onnx" in message and "not installed" in message:
            raise ExportDependencyError(
                "onnx is not installed; run: pip install onnx"
            ) from exc
        raise

    return out


def build_model_for_export(
    model_name: str = "distribai-tiny",
    *,
    architecture: str | None = None,
    vocab_size: int = 256,
    seq_len: int = 32,
) -> nn.Module:
    """Construct a small native model suitable for export smoke tests."""
    from worker.src.compute.distribai_models import CustomModelBuilder, DistribAIModelWrapper

    if architecture:
        return CustomModelBuilder.create_custom_model(
            dim=32,
            n_layers=2,
            n_heads=2,
            ffn_dim=64,
            vocab_size=vocab_size,
            seq_len=seq_len,
            family=architecture,
            architecture=architecture,
        )
    if model_name in DistribAIModelWrapper.MODEL_CONFIGS:
        return DistribAIModelWrapper(model_name, vocab_size=vocab_size, seq_len=seq_len)
    return CustomModelBuilder.create_tiny_model(vocab_size=vocab_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export DistribAI model weights")
    parser.add_argument(
        "--model",
        default="distribai-tiny",
        help="Named profile (e.g. distribai-tiny) or ignored when --architecture is set",
    )
    parser.add_argument(
        "--architecture",
        default=None,
        help="Optional family override (dense_ffn, gru, decoder_transformer, …)",
    )
    parser.add_argument(
        "--format",
        choices=("safetensors", "onnx"),
        required=True,
        help="Export format",
    )
    parser.add_argument("--out", required=True, help="Output file path")
    parser.add_argument("--seq-len", type=int, default=8, help="Dummy sequence length for ONNX")
    parser.add_argument("--vocab-size", type=int, default=256)
    args = parser.parse_args(argv)

    try:
        model = build_model_for_export(
            args.model,
            architecture=args.architecture,
            vocab_size=args.vocab_size,
            seq_len=max(args.seq_len, 8),
        )
        if args.format == "safetensors":
            path = export_safetensors(model, args.out)
        else:
            path = export_onnx(
                model,
                args.out,
                seq_len=args.seq_len,
                vocab_hint=args.vocab_size,
            )
    except ExportDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
