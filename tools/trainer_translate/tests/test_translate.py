"""Tests for the Python -> C++/LibTorch trainer translator.

Covers the two contracts that matter:

  * a normal trainer translates into a runnable job (and the LibTorch runner
    actually trains it when the binary is built); and
  * everything else fails CLOSED: no job directory, non-zero status, and an
    operator notice naming the missing feature or the script bug.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from trainer_translate import TranslationError
from trainer_translate import translate_trainer as translate
from trainer_translate.notices import ISSUES_URL, TOOL_PATH, bugfix_notice, unsupported_notice

ROOT = Path(__file__).resolve().parents[3]
TORCH_BIN = ROOT / "build" / "cpp_port" / "distribai_torch_train"

SUPPORTED = """
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, n_in=6, n_hidden=16, n_out=1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_out))

    def forward(self, x):
        return self.net(x)


def main():
    model = MLP()
    loader = DataLoader(TensorDataset(torch.randn(64, 6), torch.randn(64, 1)), batch_size=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    crit = nn.MSELoss()
    for epoch in range(5):
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()


if __name__ == "__main__":
    main()
"""

UNSUPPORTED_OPTIMIZER = """
import torch
import torch.nn as nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


def main():
    model = Tiny()
    # Rprop is a real optimizer but the LibTorch job spec does not support it.
    opt = torch.optim.Rprop(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)
    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    opt.step()


if __name__ == "__main__":
    main()
"""

BUGGY_SCRIPT = """
import torch


def main():
    # An obvious script bug: shape mismatch the translator surfaces verbatim.
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    raise RuntimeError("boom: " + str((a @ b.t()).shape))


if __name__ == "__main__":
    main()
"""

NO_MODEL = """
import torch
import torch.nn as nn


def build_only():
    # No module is instantiated when the script runs, so there is nothing to translate.
    return nn.Linear(4, 2)


if __name__ == "__main__":
    build_only()
"""


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


def test_supported_trainer_translates_to_job(tmp_path):
    trainer = _write(tmp_path, "trainer.py", SUPPORTED)
    out = tmp_path / "job"

    res = translate(trainer, out, steps=20, sandbox=False)

    assert res.optimizer == "adamw"
    assert res.loss == "mse"
    assert res.input_shape == [6]
    assert res.target_shape == [1]
    assert (out / "model.pt").is_file()
    assert (out / "job.json").is_file()

    spec = json.loads((out / "job.json").read_text())
    assert spec["optimizer"] == "adamw"
    assert spec["lr"] == pytest.approx(1e-3)
    assert spec["weight_decay"] == pytest.approx(0.01)
    assert spec["translation"]["tool"] == TOOL_PATH


@pytest.mark.skipif(not TORCH_BIN.is_file(), reason="LibTorch runner not built (make torch)")
def test_translated_job_runs_on_libtorch(tmp_path):
    trainer = _write(tmp_path, "trainer.py", SUPPORTED)
    out = tmp_path / "job"
    res = translate(trainer, out, steps=15, sandbox=False)

    proc = subprocess.run(
        [str(TORCH_BIN), "--spec", str(out / "job.json"), "--json", "--no-sandbox"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["engine"] == "libtorch"
    assert payload["grad_len"] > 0
    assert res.optimizer == payload["optimizer"]


def test_unsupported_optimizer_fails_closed(tmp_path):
    trainer = _write(tmp_path, "trainer.py", UNSUPPORTED_OPTIMIZER)
    out = tmp_path / "job"

    with pytest.raises(TranslationError) as exc:
        translate(trainer, out, steps=5, sandbox=False)

    assert exc.value.report.status == "unsupported"
    assert any("Rprop" in f for f in exc.value.report.missing_features)
    # Fail-closed: nothing emitted.
    assert not (out / "model.pt").exists()
    assert not (out / "job.json").exists()

    notice = unsupported_notice(exc.value.report)
    assert "LibTorch is the only live train path" in notice
    assert "PRs" in notice
    assert ISSUES_URL in notice


def test_buggy_script_reports_bugfix_not_unsupported(tmp_path):
    trainer = _write(tmp_path, "trainer.py", BUGGY_SCRIPT)
    out = tmp_path / "job"

    with pytest.raises(TranslationError) as exc:
        translate(trainer, out, steps=5, sandbox=False)

    assert exc.value.report.status == "needs_bugfix"
    assert "boom" in (exc.value.report.bug_summary or "")
    assert not out.exists() or not any(out.iterdir())

    notice = bugfix_notice(exc.value.report)
    assert "BUG FIX" in notice
    assert "takes a while" in notice


def test_no_model_fails_closed(tmp_path):
    trainer = _write(tmp_path, "trainer.py", NO_MODEL)
    out = tmp_path / "job"

    with pytest.raises(TranslationError) as exc:
        translate(trainer, out, steps=5, sandbox=False)

    assert exc.value.report.status == "unsupported"
    assert exc.value.report.missing_features


def test_input_shape_override_is_honored(tmp_path):
    # A Conv model has no inferable spatial shape; the caller can supply it.
    conv = """
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=(2, 3)))


def main():
    model = ConvNet()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    crit = nn.MSELoss()
    x = torch.randn(2, 3, 8, 8)
    y = torch.randn(2, 2)
    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    opt.step()


if __name__ == "__main__":
    main()
"""
    trainer = _write(tmp_path, "trainer.py", conv)
    out = tmp_path / "job"
    res = translate(trainer, out, steps=5, sandbox=False, input_shape="3,8,8", target_shape="2")

    spec = json.loads((out / "job.json").read_text())
    assert spec["input_shape"] == [3, 8, 8]
    assert spec["target_shape"] == [2]
    assert res.optimizer == "sgd"
