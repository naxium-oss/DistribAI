#!/usr/bin/env python3
"""Fail CI when Bandit reports HIGH-severity issues above the allowed threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enforce Bandit HIGH severity gate")
    parser.add_argument("--report", required=True, help="Path to bandit JSON report")
    parser.add_argument(
        "--max-high",
        type=int,
        default=0,
        help="Maximum allowed HIGH findings (default: 0)",
    )
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"bandit gate: missing report {report_path}", file=sys.stderr)
        return 1

    data = json.loads(report_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    high = [item for item in results if item.get("issue_severity") == "HIGH"]

    if len(high) > args.max_high:
        print(
            f"bandit gate: {len(high)} HIGH issue(s) exceed max {args.max_high}",
            file=sys.stderr,
        )
        for item in high[:20]:
            loc = item.get("filename", "?")
            line = item.get("line_number", "?")
            test_id = item.get("test_id", "?")
            print(f"  - {loc}:{line} {test_id}", file=sys.stderr)
        return 1

    print(f"bandit gate: ok ({len(high)} HIGH, threshold {args.max_high})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
