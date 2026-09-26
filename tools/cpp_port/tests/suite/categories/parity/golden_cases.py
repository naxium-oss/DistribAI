#!/usr/bin/env python3
"""Golden parity case matrix (suite 1): 50 cases across seeds, model sizes,
steps, sandbox multiplicity and AdamW settings.

Same-seed contracts evaluated per case (source: runtime/baselines/parity_matrix.md):
  same_seed_digest_identical   two runner instances, same seed -> identical digest
  same_seed_losses_identical   full loss trajectories bit-equal (stricter than digest)
  init_w1_first8_stable        first 8 weight bits stable across instances
  grad_len_matches_params      grad vector length == param count
  final_loss_finite            loss is finite and non-negative
Determinism across cases: distinct seeds must NOT collide (seed separation).

The C++ runner (golden_cases.cpp) prints one JSON line per --case invocation:
  {"case": N, "digest": "hex8", "losses": [...], "grad_len": K, "grad_first3": [...],
   "init_w1_first8": [...], "params": P, "final_loss": x, "loss_first": x}

Usage: python3 golden_cases.py [--quick] [--report PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[6]
BIN = REPO / "build" / "cpp_port" / "golden_cases"
LIM = REPO / "tools" / "bench" / "run_limited.sh"

# AdamW settings grid: lr x wd (beta/eps stay at torch defaults in the port).
ADAM_GRID = [
    {"lr": 0.01, "wd": 0.01},  # torch defaults (parity contract row 2.x)
    {"lr": 0.001, "wd": 0.0},  # train_scaling style (no decay)
    {"lr": 0.05, "wd": 0.05},  # aggressive
    {"lr": 0.01, "wd": 0.0},  # lr default, decay off
]

# Model sizes: hidden units for the 1-H-H-1 family (1-param handled by the
# width=0 special case that mirrors Python train_scaling's 1param row).
# 30 -> 1,021 params (train_scaling "1k"), 64 -> 4,353 (bench-family 10-30-30-1
# sibling shape), 90 -> 8,461 ("10k"), 290 -> 85,261 ("100k").
SIZES = [
    {"name": "1param", "hidden": 0},
    {"name": "1k", "hidden": 30},
    {"name": "10k", "hidden": 90},
    {"name": "100k", "hidden": 290},
]

SEEDS = [42, 43, 44, 7, 999]


def build_cases() -> list[dict]:
    """50 cases: grid over sizes x seeds x AdamW settings + sandbox/repeat axes."""
    cases: list[dict] = []

    def add(hidden: int, seed: int, steps: int, adam: int, sandbox: int, tag: str):
        cases.append(
            {
                "id": len(cases),
                "hidden": hidden,
                "seed": seed,
                "steps": steps,
                "adam": adam,  # index into ADAM_GRID
                "sandbox": sandbox,  # 1x sandboxed, 2x = two sandboxes, case summed
                "tag": tag,
            }
        )

    def gen():
        n = 0
        # A) 4 sizes x 4 seeds, default AdamW, 120 steps = 16
        for s in SIZES:
            for seed in [42, 43, 44, 7]:
                yield {
                    "hidden": s["hidden"],
                    "seed": seed,
                    "steps": 120,
                    "adam": 0,
                    "sandbox": 1,
                    "tag": f"A:{s['name']}",
                }
                n += 1
        # B) 1k size x 5th seed x 4 AdamW settings = 4
        for a in range(4):
            yield {
                "hidden": 30,
                "seed": 999,
                "steps": 120,
                "adam": a,
                "sandbox": 1,
                "tag": "B:adamw",
            }
            n += 1
        # C) steps ladder on 10k: 1, 5, 60, 300 = 4
        for st in [1, 5, 60, 300]:
            yield {"hidden": 90, "seed": 42, "steps": st, "adam": 0, "sandbox": 1, "tag": "C:steps"}
            n += 1
        # D) sandbox multiplicity: 1x and 2x on 1k and 10k = 4
        for hidden in [30, 90]:
            for sb in [1, 2]:
                yield {
                    "hidden": hidden,
                    "seed": 43,
                    "steps": 80,
                    "adam": 0,
                    "sandbox": sb,
                    "tag": "D:sandbox",
                }
                n += 1
        # E) AdamW grid x sizes (10k with each setting) = 4
        for a in range(4):
            yield {
                "hidden": 90,
                "seed": 44,
                "steps": 100,
                "adam": a,
                "sandbox": 1,
                "tag": "E:adamw-size",
            }
            n += 1
        # F) cross-size seed-separation checks (same seed, different sizes) = 6
        for hidden in [30, 90, 290]:
            yield {
                "hidden": hidden,
                "seed": 42,
                "steps": 50,
                "adam": 0,
                "sandbox": 1,
                "tag": "F:sep",
            }
            n += 1
            yield {
                "hidden": hidden,
                "seed": 43,
                "steps": 50,
                "adam": 0,
                "sandbox": 1,
                "tag": "F:sep",
            }
            n += 1
        # G) 100k + non-default AdamW + long-ish steps = 4
        for a, st in [(1, 100), (2, 40), (3, 100), (0, 200)]:
            yield {
                "hidden": 290,
                "seed": 7,
                "steps": st,
                "adam": a,
                "sandbox": 1,
                "tag": "G:100k-mix",
            }
            n += 1
        # H) 1param degenerate model = 4 (steps/AdamW variations)
        for a, st in [(0, 60), (1, 60), (2, 20), (3, 120)]:
            yield {"hidden": 0, "seed": 42, "steps": st, "adam": a, "sandbox": 1, "tag": "H:1param"}
            n += 1
        # I) extras: 2x sandbox at 100k, 1-step 100k, long 100k, adam-3 at 10k = 4
        for hidden, seed, st, a, sb in [
            (290, 43, 60, 0, 2),
            (290, 999, 1, 0, 1),
            (290, 999, 300, 0, 1),
            (90, 7, 120, 3, 1),
        ]:
            yield {
                "hidden": hidden,
                "seed": seed,
                "steps": st,
                "adam": a,
                "sandbox": sb,
                "tag": "I:extra",
            }
            n += 1
        assert n == 50, n

    for c in gen():
        add(c["hidden"], c["seed"], c["steps"], c["adam"], c["sandbox"], c["tag"])
    return cases


def run_case(c: dict, quick: bool = False) -> dict:
    """Run one case twice; return per-case results + contract evaluations."""
    steps = c["steps"]
    if quick and steps > 120:
        steps = 120
    cmd = [
        str(BIN),
        "--case",
        str(c["id"]),
        "--hidden",
        str(c["hidden"]),
        "--seed",
        str(c["seed"]),
        "--steps",
        str(steps),
        "--adam",
        str(c["adam"]),
        "--sandbox",
        str(c["sandbox"]),
    ]
    out1 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out2 = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out1.returncode != 0 or out2.returncode != 0:
        return {
            "case": c["id"],
            "tag": c["tag"],
            "ok": False,
            "error": (out1.stderr or out2.stderr or "runner failed").strip()[-400:],
        }
    j1 = json.loads(out1.stdout.strip().splitlines()[-1])
    j2 = json.loads(out2.stdout.strip().splitlines()[-1])

    contracts = {
        "same_seed_digest_identical": j1["digest"] == j2["digest"],
        "same_seed_losses_identical": j1["losses"] == j2["losses"],
        "init_w1_first8_stable": j1["init_w1_first8"] == j2["init_w1_first8"],
        "grad_len_matches_params": j1["grad_len"] == j1["params"],
        "final_loss_finite": j1["final_loss"] == j1["final_loss"] and j1["final_loss"] >= 0.0,
    }
    return {
        "case": c["id"],
        "tag": c["tag"],
        "hidden": c["hidden"],
        "seed": c["seed"],
        "steps": steps,
        "adam": ADAM_GRID[c["adam"]],
        "sandbox": c["sandbox"],
        "digest": j1["digest"],
        "params": j1["params"],
        "final_loss": j1["final_loss"],
        "loss_first": j1["loss_first"],
        "grad_len": j1["grad_len"],
        "grad_first3": j1["grad_first3"],
        "contracts": contracts,
        "ok": all(contracts.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", default=str(REPO / "runtime/baselines/golden_suite_report.json"))
    args = ap.parse_args()

    cases = build_cases()
    report: dict[str, Any] = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_cases": len(cases),
        "contracts": [
            "same_seed_digest_identical",
            "same_seed_losses_identical",
            "init_w1_first8_stable",
            "grad_len_matches_params",
            "final_loss_finite",
        ],
        "cases": [],
    }
    failures = 0
    t0 = time.time()
    for c in cases:
        r = run_case(c, args.quick)
        report["cases"].append(r)
        if not r["ok"]:
            failures += 1
            print(
                f"case {c['id']:2d} {c['tag']:12s} FAIL: "
                f"{[k for k, v in r.get('contracts', {}).items() if not v] or r.get('error')}"
            )
        else:
            print(
                f"case {c['id']:2d} {c['tag']:12s} ok  digest={r['digest']} "
                f"params={r['params']} final={r['final_loss']:.6f}"
            )
    report["failures"] = failures
    report["all_ok"] = failures == 0
    report["wall_s"] = round(time.time() - t0, 1)

    # Seed-separation cross-check: same seed at different widths may share
    # nothing; different seeds at the same width must differ.
    sep_breaks = 0
    by_width: dict[int, list[dict]] = {}
    for r in report["cases"]:  # type: ignore[attr-defined]
        if r.get("tag", "").startswith("F:"):
            by_width.setdefault(r["hidden"], []).append(r)
    for rows in by_width.values():
        digests = [r["digest"] for r in rows]
        if len(set(digests)) != len(digests):
            sep_breaks += 1
    report["seed_separation_breaks"] = sep_breaks
    if sep_breaks:
        report["all_ok"] = False

    dest = Path(args.report)
    dest.write_text(json.dumps(report, indent=1))
    print(
        f"\ncases: {len(cases)}, failures: {failures}, "
        f"seed-separation breaks: {sep_breaks}, wall {report['wall_s']}s"
    )
    print(f"saved -> {dest}")
    print(f"GOLDEN-50 GATE: {'PASS' if report['all_ok'] else 'FAIL'}")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
