#!/usr/bin/env python3
"""One-command newcomer slice: unit phase + rehearsal preflight (no live orchestrator)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _resolve_python() -> str:
    for rel in (
        Path("venv/Scripts/python.exe"),
        Path("venv/bin/python3"),
        Path("venv/bin/python"),
    ):
        candidate = _ROOT / rel
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def main() -> int:
    py = _resolve_python()
    steps = [
        [py, "-m", "pytest", "tests/unit/test_preflight_and_failure_codes.py", "-q"],
        [py, str(_ROOT / "scripts" / "dev" / "rehearse_sandbox.py")],
    ]
    for cmd in steps:
        print("+", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(_ROOT))
        if rc != 0:
            print("newcomer_test: FAIL", file=sys.stderr)
            return rc
    print("newcomer_test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
