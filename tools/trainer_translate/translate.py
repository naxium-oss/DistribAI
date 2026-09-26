"""translate.py: turn a Python trainer into a C++/LibTorch grid job.

Usage::

    python -m trainer_translate.translate trainer.py --out jobs/my-job [options]

On success it writes a runnable job directory::

    jobs/my-job/
      model.pt    TorchScript architecture (captured from the trainer)
      job.json    spec for tools/cpp_port/torch/torch_train.cpp
      x.bin/y.bin data snapshot (when the trainer used a DataLoader)
      notes.md    provenance + warnings

On failure it emits **nothing** to ``--out`` and exits 3 with a structured
report plus an operator notice (see ``notices.py``). Translation is fail-closed
by design: a partially translated job is worse than a clear error.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import TranslationError, TranslationReport
from .notices import ISSUES_URL, TOOL_PATH, render_report

ROOT = Path(__file__).resolve().parents[2]
RUN_LIMITED = ROOT / "tools" / "bench" / "run_limited.sh"


@dataclass
class TranslateResult:
    job_dir: str
    job_id: str
    model_class: str
    optimizer: str
    loss: str
    steps: int
    input_shape: list[int]
    target_shape: list[int]
    target_dtype: str
    warnings: list[str] = field(default_factory=list)
    data: str = "synthetic"
    framework: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "job_dir": self.job_dir,
            "job_id": self.job_id,
            "model_class": self.model_class,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "steps": self.steps,
            "input_shape": self.input_shape,
            "target_shape": self.target_shape,
            "target_dtype": self.target_dtype,
            "data": self.data,
            "framework": self.framework,
            "warnings": list(self.warnings),
        }


def _run_capture(
    trainer: Path,
    work: Path,
    python: str,
    max_batches: int,
    input_shape: str,
    target_shape: str,
    sandbox: bool,
) -> dict[str, Any]:
    """Run the capture harness (optionally under run_limited.sh) and read back."""
    cmd = [
        python,
        "-m",
        "trainer_translate.capture",
        "--trainer",
        str(trainer),
        "--out",
        str(work),
        "--max-batches",
        str(max_batches),
    ]
    if input_shape:
        cmd += ["--input-shape", input_shape]
    if target_shape:
        cmd += ["--target-shape", target_shape]
    if sandbox and RUN_LIMITED.is_file():
        cmd = ["bash", str(RUN_LIMITED), *cmd]

    env = dict(os.environ)
    env["DISTRIBAI_TRANSLATE_STEPS"] = env.get("DISTRIBAI_TRANSLATE_STEPS", "1")
    env["PYTHONPATH"] = str(ROOT / "tools") + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    capture_json = work / "capture.json"
    if not capture_json.is_file():
        raise TranslationError(
            TranslationReport(
                status="internal",
                stage="capture",
                trainer=str(trainer),
                detail="capture harness produced no capture.json",
                traceback=(proc.stderr or proc.stdout or "")[-4000:],
            )
        )
    try:
        return json.loads(capture_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TranslationError(
            TranslationReport(
                status="internal",
                stage="capture",
                trainer=str(trainer),
                detail=f"capture.json is not valid JSON: {exc}",
            )
        ) from exc


def _report_from_capture(cap: dict[str, Any], trainer: Path) -> TranslationReport:
    status = cap.get("status", "unsupported")
    if status not in {"unsupported", "needs_bugfix", "internal"}:
        status = "unsupported"
    extras: dict[str, Any] = {}
    if cap.get("framework"):
        extras["framework"] = cap["framework"]
    return TranslationReport(
        status=status,
        stage=cap.get("stage", "capture"),
        missing_features=list(cap.get("missing_features") or []),
        bug_summary=cap.get("bug_summary"),
        traceback=cap.get("traceback"),
        detail=cap.get("detail"),
        trainer=str(trainer),
        extras=extras,
    )


def translate(
    trainer: Path,
    out: Path,
    *,
    steps: int = 200,
    device: str = "auto",
    job_id: str | None = None,
    python: str | None = None,
    max_batches: int = 8,
    input_shape: str = "",
    target_shape: str = "",
    sandbox: bool = True,
    force: bool = False,
) -> TranslateResult:
    trainer = trainer.resolve()
    if not trainer.is_file():
        raise TranslationError(
            TranslationReport(
                status="needs_bugfix",
                stage="import",
                trainer=str(trainer),
                detail=f"trainer not found: {trainer}",
            )
        )
    python = python or sys.executable

    out = out.resolve()
    if out.exists() and any(out.iterdir()) and not force:
        raise TranslationError(
            TranslationReport(
                status="internal",
                stage="emit",
                trainer=str(trainer),
                detail=f"output dir {out} is not empty (pass --force to overwrite)",
            )
        )

    work = out.parent / f".{out.name}.translate-{os.getpid()}"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    try:
        cap = _run_capture(trainer, work, python, max_batches, input_shape, target_shape, sandbox)
        if not cap.get("ok"):
            raise TranslationError(_report_from_capture(cap, trainer))

        jid = job_id or out.name or "torch-job"
        spec: dict[str, Any] = {
            "job_id": jid,
            "model": "model.pt",
            "device": device,
            "optimizer": cap["optimizer"],
            "lr": float(cap.get("defaults", {}).get("lr", 0.01) or 0.01),
            "weight_decay": float(cap.get("defaults", {}).get("weight_decay", 0.0) or 0.0),
            "momentum": float(cap.get("defaults", {}).get("momentum", 0.9) or 0.9),
            "loss": cap["loss"],
            "target_dtype": cap["target_dtype"],
            "steps": int(steps),
            "batch_size": int(cap.get("batch_size") or 64),
            "seed": 42,
            "input_shape": list(cap["input_shape"]),
            "target_shape": list(cap["target_shape"]),
            "aggregate": "trimmed_mean",
            "sandbox_count": 2,
            "rlimits": {"mem_mb": 8192, "cpu_sec": 300},
            "framework": cap.get("framework"),
            "translation": {
                "tool": TOOL_PATH,
                "source_script": str(trainer),
                "model_class": cap.get("model_class"),
                "data": cap.get("data_note"),
                "warnings": list(cap.get("warnings") or []),
                "observed_steps": cap.get("steps_observed"),
            },
        }
        if (work / "x.bin").is_file() and (work / "y.bin").is_file():
            spec["input"] = "x.bin"
            spec["target"] = "y.bin"

        # Only now, with a fully built job, do we touch --out.
        out.mkdir(parents=True, exist_ok=True)
        for name in ("model.pt", "x.bin", "y.bin"):
            src = work / name
            if src.is_file():
                shutil.copy2(src, out / name)
        (out / "job.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        (out / "notes.md").write_text(_notes_md(trainer, spec, cap), encoding="utf-8")

        return TranslateResult(
            job_dir=str(out),
            job_id=jid,
            model_class=str(cap.get("model_class")),
            optimizer=str(cap["optimizer"]),
            loss=str(cap["loss"]),
            steps=int(steps),
            input_shape=list(cap["input_shape"]),
            target_shape=list(cap["target_shape"]),
            target_dtype=str(cap["target_dtype"]),
            warnings=list(cap.get("warnings") or []),
            data=str(cap.get("data_note") or "synthetic"),
            framework=cap.get("framework"),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _notes_md(trainer: Path, spec: dict[str, Any], cap: dict[str, Any]) -> str:
    warnings = cap.get("warnings") or []
    lines = [
        f"# LibTorch job: {spec['job_id']}",
        "",
        f"- Source trainer: `{trainer}`",
        f"- Model class: `{cap.get('model_class')}`",
        f"- Optimizer: `{spec['optimizer']}` (lr={spec['lr']}, weight_decay={spec['weight_decay']})",
        f"- Loss: `{spec['loss']}` / target dtype `{spec['target_dtype']}`",
        f"- Data: {cap.get('data_note')}",
        f"- Translated by `{TOOL_PATH}`. LibTorch is the only live train path.",
    ]
    fw = spec.get("framework")
    if fw:
        lines += [
            f"- Framework: `{fw.get('display')}` (`{fw.get('key')}`), adapted to a "
            f"single rank for capture ({fw.get('mode')}); detected via "
            f"{fw.get('detected_via')}.",
        ]
    lines += [
        "",
        "Run it:",
        "",
        "```bash",
        "tools/bench/run_limited.sh build/cpp_port/distribai_torch_train --spec "
        + str(spec["job_id"])
        + "/job.json --json",
        "```",
        "",
    ]
    if warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in warnings]
        lines.append("")
    lines += [
        "## Need a feature?",
        "",
        f"If this job needs something the translator does not support, ask a developer or an "
        f"AI agent to add it in `{TOOL_PATH}/`. PRs are suggested so others get the feature too; "
        f"otherwise open an issue at {ISSUES_URL} (it will be patched, but takes a while).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Translate a Python trainer into a LibTorch job")
    ap.add_argument("trainer", help="path to the Python trainer script")
    ap.add_argument("--out", required=True, help="output job directory")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--job-id", default=None)
    ap.add_argument("--python", default=None, help="interpreter with torch installed")
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--input-shape", default="", help="e.g. 3,32,32 (override)")
    ap.add_argument("--target-shape", default="", help="e.g. 10 (override)")
    ap.add_argument("--no-sandbox", action="store_true", help="skip run_limited.sh for capture")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    ap.add_argument("--json", action="store_true", help="machine-readable result on stdout")
    args = ap.parse_args(argv)

    try:
        res = translate(
            Path(args.trainer),
            Path(args.out),
            steps=args.steps,
            device=args.device,
            job_id=args.job_id,
            python=args.python,
            max_batches=args.max_batches,
            input_shape=args.input_shape,
            target_shape=args.target_shape,
            sandbox=not args.no_sandbox,
            force=args.force,
        )
    except TranslationError as exc:
        report = exc.report
        print(render_report(report), file=sys.stderr)
        if args.json:
            print(json.dumps({"status": "failed", **report.to_dict()}, indent=2))
        else:
            print(json.dumps({"status": "failed", **report.to_dict()}, indent=2), file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(f"translated {res.model_class} -> {res.job_dir}")
        print(f"  optimizer={res.optimizer} loss={res.loss} steps={res.steps} data={res.data}")
        if res.framework:
            print(f"  framework={res.framework.get('display')} ({res.framework.get('mode')})")
        print(f"  run: build/cpp_port/distribai_torch_train --spec {res.job_dir}/job.json --json")
        for w in res.warnings:
            print(f"  warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
