#!/usr/bin/env python3
"""Apply feature-group commits on review-security-patches-and-updates-may-2026."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.dev.feature_commit_groups import COMMIT_GROUPS, EXCLUDE_PREFIXES

ROOT = Path(__file__).resolve().parents[2]
TARGET_BRANCH = "review-security-patches-and-updates-may-2026"
SOURCE_BRANCH = "distributed-training-coordinator-dashboard-security"
STATE_FILE = ROOT / "runtime" / "feature_commit_state.txt"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def path_matches_group(rel: str, prefixes: list[str]) -> bool:
    for prefix in prefixes:
        if rel == prefix.rstrip("/") or rel.startswith(prefix):
            return True
    return False


def is_excluded(rel: str) -> bool:
    return any(rel == p.rstrip("/") or rel.startswith(p) for p in EXCLUDE_PREFIXES)


def list_changed_paths() -> list[str]:
    result = run("git", "status", "--porcelain", check=True)
    paths: set[str] = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        entry = line[3:]
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        rel = entry.replace("\\", "/").strip()
        if rel and not is_excluded(rel):
            paths.add(rel)
    return sorted(paths)


def assign_paths_to_groups(all_paths: list[str]) -> dict[int, list[str]]:
    assigned: dict[int, list[str]] = {i: [] for i in range(len(COMMIT_GROUPS))}
    leftover: list[str] = []
    for rel in all_paths:
        if "__pycache__" in rel or rel.endswith(".pyc"):
            assigned[0].append(rel)
            continue
        matched = False
        for idx, (_, prefixes) in enumerate(COMMIT_GROUPS):
            if path_matches_group(rel, prefixes):
                assigned[idx].append(rel)
                matched = True
                break
        if not matched:
            leftover.append(rel)
    if leftover:
        assigned[len(COMMIT_GROUPS) - 1].extend(leftover)
    return assigned


def stage_paths(paths: list[str]) -> None:
    if not paths:
        return
    status = run("git", "status", "--porcelain", check=True).stdout.splitlines()
    candidates: set[str] = set()
    for line in status:
        if not line.strip():
            continue
        entry = line[3:]
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        rel = entry.replace("\\", "/").strip()
        for prefix in paths:
            if rel == prefix.rstrip("/") or rel.startswith(prefix):
                candidates.add(rel)
                break
    for rel in sorted(candidates):
        add = run("git", "add", "-A", "--", rel, check=False)
        if add.returncode != 0:
            run("git", "add", "-u", "--", rel, check=False)


def read_state() -> int:
    if not STATE_FILE.exists():
        return 0
    try:
        return int(STATE_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return 0


def write_state(index: int) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(str(index), encoding="utf-8")


def prepare_branch() -> None:
    status = run("git", "status", "--porcelain", check=True)
    has_wip = bool(status.stdout.strip())
    if has_wip:
        run(
            "git",
            "stash",
            "push",
            "-u",
            "-m",
            "split-feature-commits-wip",
            check=True,
        )

    run("git", "branch", f"backup/pre-split-{SOURCE_BRANCH}", SOURCE_BRANCH, check=False)
    branches = run("git", "branch", "--list", TARGET_BRANCH, check=True).stdout
    if TARGET_BRANCH not in branches:
        run("git", "branch", TARGET_BRANCH, "main", check=True)
    run("git", "checkout", TARGET_BRANCH, check=True)
    run("git", "reset", "--hard", "main", check=True)
    run("git", "checkout", SOURCE_BRANCH, "--", ".", check=True)

    if has_wip:
        pop = run("git", "stash", "pop", check=False)
        if pop.returncode != 0:
            print(pop.stdout)
            print(pop.stderr, file=sys.stderr)
            raise SystemExit("stash pop failed — resolve conflicts manually")

    # Leave paths unstaged; split commits stage per group only.


def commit_one(index: int, assigned: dict[int, list[str]]) -> bool:
    if index >= len(COMMIT_GROUPS):
        return False
    message, _ = COMMIT_GROUPS[index]
    paths = assigned.get(index, [])
    if not paths:
        write_state(index + 1)
        return True
    stage_paths(paths)
    diff = run("git", "diff", "--cached", "--quiet", check=False)
    if diff.returncode == 0:
        print(f"skip empty group {index}: {message}")
        write_state(index + 1)
        return True
    run("git", "commit", "-m", message, check=True)
    print(f"committed group {index}: {message} ({len(paths)} paths)")
    write_state(index + 1)
    return True


def finalize() -> None:
    remaining = list_changed_paths()
    if remaining:
        stage_paths(remaining)
        diff = run("git", "diff", "--cached", "--quiet", check=False)
        if diff.returncode != 0:
            run(
                "git",
                "commit",
                "-m",
                "chore: remaining split paths and pycache deletions",
                check=True,
            )
            print(f"committed final catch-all ({len(remaining)} paths)")
    leftover = [rel for rel in list_changed_paths() if not rel.startswith("external/mytrainer")]
    if leftover:
        print("WARNING: uncommitted paths remain:")
        for rel in leftover:
            print(f"  {rel}")
        raise SystemExit(1)
    print("split complete; working tree clean")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--next", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        idx = read_state()
        total = len(COMMIT_GROUPS)
        print(f"progress: {idx}/{total}")
        if idx < total:
            print(f"next: {COMMIT_GROUPS[idx][0]}")
        return

    if args.prepare:
        prepare_branch()
        write_state(0)
        print("prepared branch and reset state")
        return

    assigned = assign_paths_to_groups(list_changed_paths())
    idx = read_state()

    if args.next:
        if idx >= len(COMMIT_GROUPS):
            print("all groups committed")
            return
        commit_one(idx, assigned)
        return

    if args.all:
        prepare_branch()
        assigned = assign_paths_to_groups(list_changed_paths())
        for i in range(len(COMMIT_GROUPS)):
            commit_one(i, assigned)
        finalize()
        return

    if args.finalize:
        finalize()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
