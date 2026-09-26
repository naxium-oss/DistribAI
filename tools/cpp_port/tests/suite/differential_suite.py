#!/usr/bin/env python3
"""Differential suite (category: differential): Python harness vs C++ over a
fixed corpus.

Runs tools/cpp_port/tests/compare.py (the dual-run comparer, must-match
tolerances from parity_matrix.md) over the fixed seed corpus, then writes
runtime/baselines/diff_corpus_report.json with the corpus manifest and
per-seed verdicts. Exit 0 iff every corpus entry passes (tight or bounded
relu-threshold flips within the documented fraction gate).

Usage:
    python3 differential_suite.py                # full corpus
    python3 differential_suite.py --corpus 42,43 # subset
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
COMPARE = REPO / "tools" / "cpp_port" / "tests" / "compare.py"
DEST = REPO / "runtime" / "baselines" / "diff_corpus_report.json"
PY = REPO / ".venv" / "bin" / "python"
LIM = REPO / "tools" / "bench" / "run_limited.sh"

# Fixed differential corpus: parity seeds plus edge picks (0, 1, 2, 3 cover
# tiny-seed paths; 999/123 exercise stream-advance depth).
CORPUS = {
    "parity_mission_seeds": [42, 43, 44],
    "edge_seeds": [0, 1, 2, 3, 7, 999, 123],
}
STEPS = 200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="", help="comma list overriding the default corpus")
    args = ap.parse_args()

    if args.corpus:
        seeds = [int(s) for s in args.corpus.split(",") if s]
    else:
        seeds = CORPUS["parity_mission_seeds"] + CORPUS["edge_seeds"]

    if not PY.exists():
        print("SKIP: .venv not available (python harness required)")
        return 3

    report = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "corpus": CORPUS,
        "seeds_run": seeds,
        "steps": STEPS,
        "comparer": str(COMPARE.relative_to(REPO)),
        "runs": [],
    }
    t0 = time.time()
    cmd = [
        str(LIM),
        str(PY),
        str(COMPARE),
        "--seeds",
        ",".join(str(s) for s in seeds),
        "--steps",
        str(STEPS),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    tail = [line for line in out.stdout.splitlines() if line.strip()]
    verdict_line = tail[-1] if tail else ""
    report["comparer_verdict"] = verdict_line
    report["comparer_rc"] = out.returncode

    # compare.py writes dualrun_diffs.json; fold the per-seed verdicts in.
    dual = REPO / "runtime" / "baselines" / "dualrun_diffs.json"
    if dual.exists():
        d = json.loads(dual.read_text())
        report["runs"] = [
            {
                "seed": r.get("seed"),
                "ok": r.get("ok"),
                "classification": r.get("classification", ""),
            }
            for r in d.get("runs", [])
        ]
        report["flip_fraction"] = d.get("flip_fraction")
        report["relu_threshold_flips"] = d.get("relu_threshold_flips")

    report["all_ok"] = out.returncode == 0
    report["wall_s"] = round(time.time() - t0, 1)
    DEST.write_text(json.dumps(report, indent=1))

    runs_raw = report.get("runs")
    runs: list[dict] = [dict(r) for r in runs_raw] if isinstance(runs_raw, list) else []
    for r in runs:
        print(f"seed {r['seed']}: {'OK' if r['ok'] else 'BREAK'} {r['classification']}")
    print(f"saved -> {DEST}")
    print(f"DIFFERENTIAL GATE: {'PASS' if report['all_ok'] else 'FAIL'} ({verdict_line})")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
