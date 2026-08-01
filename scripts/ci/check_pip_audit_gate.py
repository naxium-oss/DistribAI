#!/usr/bin/env python3
"""Fail CI when pip-audit reports known vulnerabilities above threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enforce pip-audit vulnerability gate")
    parser.add_argument("--report", required=True, help="Path to pip-audit JSON report")
    parser.add_argument(
        "--max-critical",
        type=int,
        default=0,
        help="Maximum allowed CRITICAL findings (default: 0)",
    )
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"pip-audit gate: missing report {report_path}", file=sys.stderr)
        return 1

    data = json.loads(report_path.read_text(encoding="utf-8"))
    deps = data.get("dependencies", data if isinstance(data, list) else [])
    critical = []
    for dep in deps:
        for vuln in dep.get("vulns", []):
            if (vuln.get("severity") or "").upper() == "CRITICAL":
                critical.append((dep.get("name"), vuln.get("id")))

    if len(critical) > args.max_critical:
        print(
            f"pip-audit gate: {len(critical)} CRITICAL issue(s) exceed max {args.max_critical}",
            file=sys.stderr,
        )
        for name, vid in critical[:20]:
            print(f"  - {name}: {vid}", file=sys.stderr)
        return 1

    print(f"pip-audit gate: ok ({len(critical)} CRITICAL, threshold {args.max_critical})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
