"""Restore setTimeout/setInterval calls broken by automated XSS patch comments."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "worker" / "src" / "dashboard" / "static" / "node" / "index.html"


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")
    original = text

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
    text = text.replace(
        "timeout = // Safe setTimeout removedfunction() {",
        "timeout = setTimeout(function() {",
    )
    text = text.replace(
        "// Safe setTimeout removedfunction() { inThrottle = false; }, limit);",
        "setTimeout(function() { inThrottle = false; }, limit);",
    )
    text = text.replace("// Safe setTimeout removedfunction() {", "setTimeout(function() {")
    text = text.replace("// Safe setTimeout removed() => {", "setTimeout(() => {")
    text = text.replace(
        "var interval = // Safe setInterval removedfunction() {",
        "var interval = setInterval(function() {",
    )
    text = text.replace(
        "benchState.tipInterval = // Safe setInterval removed() => {",
        "benchState.tipInterval = setInterval(() => {",
    )
    text = text.replace(
        "// Safe setInterval removedcheckSessionTimeout, 60000);",
        "setInterval(checkSessionTimeout, 60000);",
    )
    text = text.replace(
        "if (el) { el.style.opacity = '0'; // Safe setTimeout removed() => { el.textContent = tip.text; el.style.opacity = '1'; }, 200); }",
        "if (el) { el.style.opacity = '0'; setTimeout(() => { el.textContent = tip.text; el.style.opacity = '1'; }, 200); }",
    )

    if text == original:
        print("no changes")
        return

    TARGET.write_text(text, encoding="utf-8")
    left_t = text.count("Safe setTimeout removed")
    left_i = text.count("Safe setInterval removed")
    print(f"updated {TARGET.relative_to(ROOT)}")
    print(f"remaining Safe setTimeout removed: {left_t}")
    print(f"remaining Safe setInterval removed: {left_i}")


if __name__ == "__main__":
    main()
