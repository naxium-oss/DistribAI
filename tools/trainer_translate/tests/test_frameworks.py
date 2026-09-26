"""Tests for the Megatron-LM / torchtitan / slime framework adapters.

The adapter contract has two halves:

  * a trainer that initializes torch.distributed through one of those frameworks
    is recognized, adapted to a single rank, and translated into a job tagged
    with the framework; and
  * a plain distributed trainer that is NOT on a supported framework still fails
    closed, unchanged.

The frameworks are not installed here, so the tests lay down tiny stand-in
packages and put them on the capture subprocess's PYTHONPATH. That exercises the
real detection and shim code without pulling in a framework.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from trainer_translate import TranslationError
from trainer_translate.errors import TranslationReport
from trainer_translate.frameworks import FRAMEWORKS, detect_framework, install_single_rank
from trainer_translate.notices import unsupported_notice
from trainer_translate.translate import translate

FRAMEWORK_TRAINER = """
import torch
import torch.nn as nn
import {module}


class LM(nn.Module):
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        return self.head(self.embed(x))


def main():
    # Every one of these frameworks brings its own process group. The adapter
    # turns this call and the collectives into single-rank no-ops.
    torch.distributed.init_process_group("{backend}")
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    model = LM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    x = torch.randint(0, 32, (8, 4))
    y = torch.randint(0, 32, (8, 4))
    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    opt.step()
    assert rank == 0 and world == 1


if __name__ == "__main__":
    main()
"""

PLAIN_DISTRIBUTED = """
import torch
import torch.nn as nn


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc(x)


def main():
    torch.distributed.init_process_group("gloo")
    model = Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
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

UNSCRIPTABLE_TRAINER = """
import torch
import torch.nn as nn
import {module}


class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        # .numpy() is not TorchScript-scriptable, so translation must fail closed
        # and the notice must still name the framework.
        return torch.from_numpy(self.fc(x).numpy())


def main():
    torch.distributed.init_process_group("nccl")
    model = Bad()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    crit = nn.MSELoss()
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    opt.zero_grad()
    loss = crit(model(x), y)
    loss.backward()
    opt.step()


if __name__ == "__main__":
    main()
"""


def _write(tmp_path: Path, name: str, src: str) -> Path:
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return p


def _install_fake_framework(tmp_path: Path, monkeypatch, module: str) -> None:
    """Drop a stand-in package on the capture subprocess's import path."""
    root = tmp_path / "fwpkgs"
    pkg = root / module
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text('__version__ = "0-test"\n', encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(root))


@pytest.mark.parametrize(
    "module,key,display,backend",
    [
        ("megatron", "megatron", "Megatron-LM", "nccl"),
        ("torchtitan", "torchtitan", "torchtitan", "nccl"),
        ("slime", "slime", "slime", "nccl"),
    ],
)
def test_framework_trainer_translates_single_rank(
    tmp_path, monkeypatch, module, key, display, backend
):
    _install_fake_framework(tmp_path, monkeypatch, module)
    trainer = _write(
        tmp_path, "trainer.py", FRAMEWORK_TRAINER.format(module=module, backend=backend)
    )
    out = tmp_path / "job"

    res = translate(trainer, out, steps=5, sandbox=False)

    assert res.framework is not None
    assert res.framework["key"] == key
    assert res.framework["display"] == display
    assert res.framework["mode"] == "single_rank"

    spec = json.loads((out / "job.json").read_text())
    assert spec["framework"]["key"] == key
    assert spec["framework"]["mode"] == "single_rank"
    # A real job still gets emitted, model and all.
    assert (out / "model.pt").is_file()
    # The single-rank adaptation is recorded as a warning, not hidden.
    assert any(display in w for w in spec["translation"]["warnings"])

    notes = (out / "notes.md").read_text()
    assert display in notes


def test_plain_distributed_trainer_still_fails_closed(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    trainer = _write(tmp_path, "trainer.py", PLAIN_DISTRIBUTED)
    out = tmp_path / "job"

    with pytest.raises(TranslationError) as exc:
        translate(trainer, out, steps=5, sandbox=False)

    assert exc.value.report.status == "unsupported"
    assert any("torch.distributed" in f for f in exc.value.report.missing_features)
    assert not (out / "job.json").exists()


def test_unscriptable_framework_model_fails_closed_with_framework_named(tmp_path, monkeypatch):
    _install_fake_framework(tmp_path, monkeypatch, "megatron")
    trainer = _write(tmp_path, "trainer.py", UNSCRIPTABLE_TRAINER.format(module="megatron"))
    out = tmp_path / "job"

    with pytest.raises(TranslationError) as exc:
        translate(trainer, out, steps=5, sandbox=False)

    report = exc.value.report
    assert report.status == "unsupported"
    assert report.extras["framework"]["key"] == "megatron"
    assert not (out / "job.json").exists()

    notice = unsupported_notice(report)
    assert "Megatron-LM" in notice
    assert "single-rank" in notice.lower()
    # The standard fail-closed guidance is still present.
    assert "PRs" in notice


def test_detect_framework_from_loaded_modules():
    assert detect_framework(loaded_modules=["megatron.core", "torch"]).key == "megatron"
    assert detect_framework(loaded_modules=["torchtitan.train"]).key == "torchtitan"
    assert detect_framework(loaded_modules=["slime.rollout"]).key == "slime"
    assert detect_framework(loaded_modules=["torch", "numpy"]) is None


def test_detect_framework_ignores_source_without_installed_package():
    # A mere mention must not reclassify a plain script.
    assert detect_framework(source="# a note about megatron.core\n") is None


def test_install_single_rank_makes_distributed_an_identity():
    torch = pytest.importorskip("torch")
    undo = install_single_rank(torch)
    try:
        assert torch.distributed.get_world_size() == 1
        assert torch.distributed.get_rank() == 0
        assert torch.distributed.is_initialized() is True
        group = torch.distributed.new_group()
        assert group.size() == 1 and group.rank() == 0
        t = torch.ones(2, 2)
        assert torch.distributed.all_reduce(t) is t
    finally:
        for u in undo:
            u()


def test_every_framework_spec_has_a_hint():
    from trainer_translate.frameworks import framework_hint

    for spec in FRAMEWORKS:
        assert framework_hint(spec.key)


def test_framework_report_round_trips_in_notice():
    report = TranslationReport(
        status="unsupported",
        stage="script",
        missing_features=["TorchScript-scriptable model"],
        trainer="trainer.py",
        extras={"framework": {"key": "slime", "display": "slime", "mode": "single_rank"}},
    )
    notice = unsupported_notice(report)
    assert "slime" in notice
