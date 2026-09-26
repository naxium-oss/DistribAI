"""Unit tests for portable weight export helpers."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from worker.src.compute.export_weights import (
    ExportDependencyError,
    build_model_for_export,
    export_onnx,
    export_safetensors,
)
from worker.src.compute.export_weights import (
    main as export_main,
)


class _TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_export_safetensors_missing_dependency(tmp_path):
    model = _TinyLinear()
    out = tmp_path / "w.safetensors"
    with patch.dict(sys.modules, {"safetensors": None, "safetensors.torch": None}):
        with pytest.raises(ExportDependencyError, match="safetensors"):
            export_safetensors(model, out)


def test_export_safetensors_with_mock_backend(tmp_path):
    model = _TinyLinear()
    out = tmp_path / "weights.safetensors"
    mock_save = MagicMock()
    mock_torch_mod = MagicMock()
    mock_torch_mod.save_file = mock_save
    with patch.dict(
        sys.modules,
        {"safetensors": MagicMock(), "safetensors.torch": mock_torch_mod},
    ):
        path = export_safetensors(model, out)
    assert path == out
    mock_save.assert_called_once()
    tensors, written = mock_save.call_args[0]
    assert "fc.weight" in tensors
    assert written == str(out)


def test_export_onnx_tiny_dense_ffn(tmp_path):
    model = build_model_for_export(
        architecture="dense_ffn",
        vocab_size=32,
        seq_len=16,
    )
    out = tmp_path / "tiny.onnx"
    path = export_onnx(model, out, batch_size=1, seq_len=4, vocab_hint=32)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_export_cli_onnx(tmp_path):
    out = tmp_path / "cli.onnx"
    code = export_main(
        [
            "--architecture",
            "dense_ffn",
            "--format",
            "onnx",
            "--out",
            str(out),
            "--seq-len",
            "4",
            "--vocab-size",
            "32",
        ]
    )
    assert code == 0
    assert out.is_file()


def test_export_cli_safetensors_reports_missing_dep(tmp_path):
    with patch.dict(sys.modules, {"safetensors": None, "safetensors.torch": None}):
        code = export_main(
            [
                "--format",
                "safetensors",
                "--out",
                str(tmp_path / "x.safetensors"),
            ]
        )
    assert code == 2
