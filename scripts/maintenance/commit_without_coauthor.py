"""Create a git commit without Cursor-injected Co-authored-by trailers.

Usage (from repo root, after staging):
  python scripts/maintenance/commit_without_coauthor.py -F path/to/message.txt
  python scripts/maintenance/commit_without_coauthor.py -m "subject" -m "body"
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _git(*args: str, cwd: Path) -> str:
    exe = r"C:\Users\ender\AppData\Local\Programs\Git\cmd\git.exe"
    result = subprocess.run(
        [exe, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-F", "--file", dest="message_file", type=Path)
    parser.add_argument("-m", "--message", action="append", default=[])
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    if args.message_file:
        # PowerShell `Set-Content -Encoding utf8` writes a BOM; utf-8-sig strips it.
        message = args.message_file.read_text(encoding="utf-8-sig").strip() + "\n"
    elif args.message:
        message = "\n\n".join(args.message)
    else:
        parser.error("provide -F or at least one -m")

    tree = _git("write-tree", cwd=root)
    try:
        parent = _git("rev-parse", "HEAD", cwd=root)
        parents = ["-p", parent]
    except SystemExit:
        parents = []

    merge_head = root / ".git" / "MERGE_HEAD"
    if merge_head.is_file():
        merge_parent = merge_head.read_text(encoding="utf-8").strip()
        if merge_parent:
            parents.extend(["-p", merge_parent])

    proc = subprocess.run(
        [
            r"C:\Users\ender\AppData\Local\Programs\Git\cmd\git.exe",
            "commit-tree",
            tree,
            *parents,
        ],
        cwd=root,
        input=message.encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.decode("utf-8", errors="replace") or proc.stdout.decode("utf-8", errors="replace"))
    new_commit = proc.stdout.decode("utf-8").strip()
    _git("reset", "--hard", new_commit, cwd=root)
    print(new_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
