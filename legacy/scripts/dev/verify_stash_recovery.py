#!/usr/bin/env python3
"""Compare working tree against recovered stash commit."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STASH = "b5e8f5ee1c1485f444bed63e44b31d54273984fa"
UNTRACKED = f"{STASH}^3"
SKIP = {
    "node_modules",
    "dist",
    ".git",
    "venv",
    ".venv",
    "external/mytrainer",
    "runtime/smoke",
    "runtime/gui-test-node",
    "runtime/triage_observe.txt",
    "runtime/triage_organize.txt",
    "runtime/triage_reflect.txt",
    "runtime/triage_strategize.json",
    "runtime/pen-orch.err",
    "bandit-report.json",
    ".desloppify",
    "test-results",
    "~",
}


def skip(rel: str) -> bool:
    return any(rel == s or rel.startswith(s + "/") for s in SKIP)


def ls_tree(ref: str) -> set[str]:
    out = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", ref], cwd=ROOT, text=True)
    return {line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()}


def disk_files() -> set[str]:
    found: set[str] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if skip(rel):
            continue
        found.add(rel)
    return found


def main() -> None:
    tracked = {p for p in ls_tree(STASH) if not skip(p)}
    untracked = {p for p in ls_tree(UNTRACKED) if not skip(p)}
    expected = tracked | untracked
    on_disk = disk_files()
    missing = sorted(expected - on_disk)
    extra = sorted(on_disk - expected - {p for p in on_disk if p.startswith("~/")})
    print(f"stash tracked: {len(tracked)}")
    print(f"stash untracked: {len(untracked)}")
    print(f"expected union: {len(expected)}")
    print(f"on disk (filtered): {len(on_disk)}")
    print(f"missing from disk: {len(missing)}")
    for rel in missing[:40]:
        print(f"  MISSING {rel}")
    if len(missing) > 40:
        print(f"  ... and {len(missing) - 40} more")
    print(f"extra on disk not in stash: {len(extra)}")
    for rel in extra[:20]:
        print(f"  EXTRA {rel}")


if __name__ == "__main__":
    main()
