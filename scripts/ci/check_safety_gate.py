#!/usr/bin/env python3
"""Fail CI when Safety reports HIGH/CRITICAL dependency vulnerabilities above threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _severity_rank(severity: str) -> int:
    s = (severity or "").upper()
    if s in ("CRITICAL", "HIGH"):
        return 2
    if s in ("MEDIUM", "MODERATE"):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enforce Safety vulnerability gate")
    parser.add_argument("--report", required=True, help="Path to safety JSON report")
    parser.add_argument(
        "--max-high",
        type=int,
        default=0,
        help="Maximum allowed HIGH+CRITICAL findings (default: 0)",
    )
    args = parser.parse_args(argv)

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"safety gate: missing report {report_path}", file=sys.stderr)
        return 1

    data = json.loads(report_path.read_text(encoding="utf-8"))
    vulns = data.get("vulnerabilities", data if isinstance(data, list) else [])
    flagged: list[tuple[str, str, str]] = []
    for item in vulns:
        severity = (
            item.get("severity")
            or item.get("vulnerability_severity")
            or item.get("cvss_severity")
            or ""
        )
        if _severity_rank(str(severity)) >= 2:
            pkg = item.get("package_name") or item.get("package") or "?"
            vid = item.get("vulnerability_id") or item.get("id") or "?"
            flagged.append((pkg, vid, str(severity)))

    if len(flagged) > args.max_high:
        print(
            f"safety gate: {len(flagged)} HIGH/CRITICAL issue(s) exceed max {args.max_high}",
            file=sys.stderr,
        )
        for pkg, vid, sev in flagged[:20]:
            print(f"  - {pkg}: {vid} ({sev})", file=sys.stderr)
        return 1

    print(f"safety gate: ok ({len(flagged)} HIGH/CRITICAL, threshold {args.max_high})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
