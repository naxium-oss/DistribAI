"""Add operator-banner.js to node dashboard HTML after shared/scripts.js."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODE_DIR = ROOT / "worker" / "src" / "dashboard" / "static" / "node"


def main() -> None:
    for path in sorted(NODE_DIR.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        if "operator-banner.js" in text:
            continue
        if "/shared/scripts.js" not in text:
            continue
        text2, count = re.subn(
            r'(<script src="/shared/scripts\.js"></script>)',
            r'\1\n    <script src="/shared/operator-banner.js"></script>',
            text,
            count=1,
        )
        if count == 0:
            text2, count = re.subn(
                r'(<script src="/shared/scripts\.js">)',
                r'\1</script>\n    <script src="/shared/operator-banner.js"></script>',
                text,
                count=1,
            )
        if count:
            path.write_text(text2, encoding="utf-8")
            print(f"patched {path.name}")


if __name__ == "__main__":
    main()
