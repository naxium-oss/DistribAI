#!/usr/bin/env python3
"""Run pytest and fail on any failure or skip."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pytest gate: 0 failed, 0 skipped")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    pytest_args = list(args.pytest_args)
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]
    if not pytest_args:
        pytest_args = ["tests", "-q", "--tb=line"]

    cmd = [sys.executable, "-m", "pytest", *pytest_args]
    print("pytest gate:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")

    skipped = 0
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in combined.splitlines():
        if " skipped" in line.lower():
            match = re.search(r"(\d+) skipped", line)
            if match:
                skipped = max(skipped, int(match.group(1)))

    if proc.returncode != 0:
        print(f"pytest gate: FAILED (exit {proc.returncode})", file=sys.stderr)
        return proc.returncode or 1
    if skipped > 0:
        print(f"pytest gate: FAILED ({skipped} skipped; require 0)", file=sys.stderr)
        return 1

    print("pytest gate: ok (0 failed, 0 skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
