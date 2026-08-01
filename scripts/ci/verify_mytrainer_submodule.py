#!/usr/bin/env python3
"""Verify external/mytrainer checkout for MyTrainer sync and training paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MYTRAINER_DIR = REPO_ROOT / "external" / "mytrainer"
CONFIG_MARKER = MYTRAINER_DIR / "configs" / "grid_architectures.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit 1 if the subtree is missing (default: warn and exit 0).",
    )
    args = parser.parse_args()

    if not MYTRAINER_DIR.is_dir():
        msg = (
            f"MyTrainer subtree not found at {MYTRAINER_DIR}. "
            "Clone or init the bundled trainer repo before using mytrainer sync:\n"
            "  git clone <mytrainer-url> external/mytrainer\n"
            "See docs/guides/mytrainer-submodule.md"
        )
        if args.require:
            print(msg, file=sys.stderr)
            return 1
        print(f"[verify_mytrainer] SKIP: {msg}")
        return 0

    if not CONFIG_MARKER.is_file():
        msg = (
            f"MyTrainer directory exists but missing {CONFIG_MARKER.name} at "
            f"{CONFIG_MARKER.parent}/. Sync expects grid architecture configs there."
        )
        if args.require:
            print(msg, file=sys.stderr)
            return 1
        print(f"[verify_mytrainer] WARN: {msg}")
        return 0

    print(f"[verify_mytrainer] OK: {MYTRAINER_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
