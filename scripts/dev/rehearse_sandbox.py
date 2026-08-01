#!/usr/bin/env python3
"""Local rehearsal: run golden template preflight + pack (no orchestrator)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.cli.distribai_cli import JobManager  # noqa: E402
from services_python.preflight import validate_script_tarball  # noqa: E402


def main() -> int:
    folder = _ROOT / "examples" / "golden_template"
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1]).resolve()
    try:
        raw, digest = JobManager.bundle_directory(folder)
    except ValueError as exc:
        print(f"rehearse_sandbox: FAIL — {exc}", file=sys.stderr)
        return 1
    ok, err, meta = validate_script_tarball(raw)
    if not ok:
        print(f"rehearse_sandbox: FAIL — preflight: {err}", file=sys.stderr)
        return 1
    print(f"rehearse_sandbox: OK folder={folder} sha256={digest} members={meta.get('member_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
