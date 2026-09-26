#!/usr/bin/env python3
"""Fail when coverage.json totals fall below the required percentage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _percent(covered: int, total: int) -> float:
    if total == 0:
        return 100.0
    return 100.0 * covered / total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enforce minimum coverage from coverage.json")
    parser.add_argument("--report", default="coverage.json")
    parser.add_argument("--min-percent", type=float, default=100.0)
    parser.add_argument("--show-missing", type=int, default=25)
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"coverage gate: missing report {report_path}", file=sys.stderr)
        return 1

    data = json.loads(report_path.read_text(encoding="utf-8"))
    totals = data.get("totals", {})
    covered = int(totals.get("covered_lines", 0))
    total = int(totals.get("num_statements", 0))
    percent = _percent(covered, total)

    if percent + 1e-9 < args.min_percent:
        print(
            f"coverage gate: {percent:.2f}% < required {args.min_percent:.2f}% "
            f"({covered}/{total} lines)",
            file=sys.stderr,
        )
        ranked: list[tuple[float, str, int, int]] = []
        for path, meta in data.get("files", {}).items():
            summary = meta.get("summary", {})
            file_total = int(summary.get("num_statements", 0))
            if file_total == 0:
                continue
            file_covered = int(summary.get("covered_lines", 0))
            ranked.append((_percent(file_covered, file_total), path, file_covered, file_total))
        ranked.sort(key=lambda row: row[0])
        for pct, path, file_covered, file_total in ranked[: args.show_missing]:
            print(f"  - {path}: {pct:.1f}% ({file_covered}/{file_total})", file=sys.stderr)
        return 1

    print(f"coverage gate: ok ({percent:.2f}% >= {args.min_percent:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
