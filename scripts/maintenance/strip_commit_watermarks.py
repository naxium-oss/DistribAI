"""Remove [Cursor] prefixes and Cursor co-author trailers from git commit messages (stdin/stdout)."""

from __future__ import annotations

import sys


def strip_message(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        body = line
        if body.startswith("[Cursor] "):
            body = body[len("[Cursor] ") :]
        lower = body.strip().lower()
        if lower.startswith("co-authored-by:") and "cursor" in lower:
            continue
        out.append(body)
    if not out and not text:
        return ""
    result = "\n".join(out)
    if text.endswith("\n"):
        result += "\n"
    return result


def main() -> None:
    sys.stdout.write(strip_message(sys.stdin.read()))


if __name__ == "__main__":
    main()
