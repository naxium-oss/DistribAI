"""Restore setTimeout/setInterval broken by automated XSS patch comments in dashboard HTML."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "worker" / "src" / "dashboard" / "static"


def fix_content(text: str) -> str:
    text = re.sub(
        r"// Safe setTimeout removed\(\)\s*=>\s*(.+?),\s*(\d+)\);",
        r"setTimeout(() => \1, \2);",
        text,
    )
    text = re.sub(
        r"(\w+)\s*=\s*// Safe setTimeout removed\(\)\s*=>\s*(.+?),\s*(\d+)\);",
        r"\1 = setTimeout(() => \2, \3);",
        text,
    )
    text = re.sub(
        r"// Safe setTimeout removed(\w+),\s*([^)]+)\);",
        r"setTimeout(\1, \2);",
        text,
    )
    text = re.sub(
        r"(\w+)\s*=\s*// Safe setTimeout removedfunction\(\)\s*\{",
        r"\1 = setTimeout(function() {",
        text,
    )
    text = re.sub(
        r"(\w+)\s*=\s*// Safe setInterval removed(\w+),\s*([^)]+)\);",
        r"\1 = setInterval(\2, \3);",
        text,
    )
    text = re.sub(
        r"// Safe setInterval removed(\w+),\s*([^)]+)\);",
        r"setInterval(\1, \2);",
        text,
    )
    replacements = [
        ("timeout = // Safe setTimeout removedfunction() {", "timeout = setTimeout(function() {"),
        (
            "// Safe setTimeout removedfunction() { inThrottle = false; }, limit);",
            "setTimeout(function() { inThrottle = false; }, limit);",
        ),
        ("// Safe setTimeout removedfunction() {", "setTimeout(function() {"),
        ("// Safe setTimeout removed() => {", "setTimeout(() => {"),
        (
            "var interval = // Safe setInterval removedfunction() {",
            "var interval = setInterval(function() {",
        ),
        (
            "benchState.tipInterval = // Safe setInterval removed() => {",
            "benchState.tipInterval = setInterval(() => {",
        ),
        (
            "benchState.timerInterval = // Safe setInterval removed() => {",
            "benchState.timerInterval = setInterval(() => {",
        ),
        (
            "benchState.timerInterval = // Safe setInterval removedupdateTimer, 1000);",
            "benchState.timerInterval = setInterval(updateTimer, 1000);",
        ),
        (
            "benchState.tipInterval = // Safe setInterval removedrotateTip, 7000);",
            "benchState.tipInterval = setInterval(rotateTip, 7000);",
        ),
        (
            "dev.pollTimer = // Safe setInterval removed() => {",
            "dev.pollTimer = setInterval(() => {",
        ),
        (
            "dev.pollTimer = // Safe setInterval removedfunction() {",
            "dev.pollTimer = setInterval(function() {",
        ),
        (
            "// Safe setInterval removedcheckSessionTimeout, 60000);",
            "setInterval(checkSessionTimeout, 60000);",
        ),
        (
            "if (el) { el.style.opacity = '0'; // Safe setTimeout removed() => { el.textContent = tip.text; el.style.opacity = '1'; }, 200); }",
            "if (el) { el.style.opacity = '0'; setTimeout(() => { el.textContent = tip.text; el.style.opacity = '1'; }, 200); }",
        ),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def main() -> None:
    total = 0
    for path in sorted(STATIC.rglob("*.html")):
        original = path.read_text(encoding="utf-8")
        updated = fix_content(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            left = updated.count("Safe setTimeout removed") + updated.count("Safe setInterval removed")
            print(f"fixed {path.relative_to(ROOT)} (remaining markers: {left})")
            total += 1
    print(f"done: {total} files updated")


if __name__ == "__main__":
    main()
