"""Extract inline script blocks from node/index.html into external JS files."""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
NODE_DIR = _REPO / "worker" / "src" / "dashboard" / "static" / "node"
HTML_PATH = NODE_DIR / "index.html"


def _collect_inline_blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    blocks: list[tuple[int, int, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"\s*<script>\s*$", line):
            start = i
            i += 1
            while i < len(lines) and "</script>" not in lines[i]:
                i += 1
            if i < len(lines):
                end = i
                js = "".join(lines[start + 1 : end])
                blocks.append((start, end, js))
                i += 1
        elif re.match(r"\s*<script\s+src=", line):
            i += 1
        else:
            i += 1
    return blocks


def main() -> None:
    content = HTML_PATH.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    orig_line_count = len(lines)

    blocks = _collect_inline_blocks(lines)
    if len(blocks) != 4:
        raise SystemExit(f"expected 4 inline script blocks, found {len(blocks)}")

    security_js = blocks[0][2].rstrip() + "\n\n" + blocks[1][2].lstrip()
    (NODE_DIR / "index-security.js").write_text(
        "/** DistribAI index preview — XSS/security helpers and error wrappers */\n"
        + security_js
        + "\n",
        encoding="utf-8",
    )
    (NODE_DIR / "index-preview.js").write_text(
        "/** DistribAI index preview — main SPA dashboard logic */\n" + blocks[2][2],
        encoding="utf-8",
    )
    (NODE_DIR / "index-dev-panel.js").write_text(
        "/** DistribAI index preview — developer panel */\n" + blocks[3][2],
        encoding="utf-8",
    )

    new_lines = list(lines)
    for start, end, _ in reversed(blocks):
        del new_lines[start : end + 1]

    # Insert external script tags (bottom-up so indices stay valid).
    # Block 4 was dev panel — before operator-banner.js
    op_banner_idx = next(
        i for i, ln in enumerate(new_lines) if "/shared/operator-banner.js" in ln
    )
    new_lines.insert(
        op_banner_idx,
        '    <script src="/index-dev-panel.js"></script>\n',
    )

    dev_comment_idx = next(
        i for i, ln in enumerate(new_lines) if "DEVELOPER PANEL JS" in ln
    )
    new_lines.insert(
        dev_comment_idx, '    <script src="/index-preview.js"></script>\n\n'
    )

    chart_idx = next(
        i for i, ln in enumerate(new_lines) if "chart.umd.min.js" in ln
    )
    new_lines.insert(chart_idx, '    <script src="/index-security.js"></script>\n')

    updated = "".join(new_lines)
    HTML_PATH.write_text(updated, encoding="utf-8")

    new_line_count = len(updated.splitlines())
    inline_left = len(re.findall(r"<script>", updated))
    print(f"Original lines: {orig_line_count}")
    print(f"New lines: {new_line_count}")
    print(f"Reduction: {orig_line_count - new_line_count}")
    print(f"Inline script tags remaining: {inline_left}")
    assert inline_left == 0, "expected no inline script tags"


if __name__ == "__main__":
    main()
