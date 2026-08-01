"""Replace duplicated dashboard headers with shared header placeholders."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODE_DIR = ROOT / "worker" / "src" / "dashboard" / "static" / "node"
ORCH_DIR = ROOT / "worker" / "src" / "dashboard" / "static" / "orch"

NODE_PAGES: dict[str, str] = {
    "admin.html": "admin",
    "benchmark.html": "benchmark",
    "credits.html": "credits",
    "dashboard.html": "dashboard",
    "dev.html": "dev",
    "help.html": "help",
    "job.html": "jobs",
    "jobs.html": "jobs",
    "settings.html": "settings",
    "thanks.html": "thanks",
}

ORCH_PAGES: dict[str, str] = {
    "orchestrator.html": "dashboard",
    "orchestrator-jobs.html": "jobs",
    "orchestrator-nodes.html": "nodes",
    "orchestrator-node.html": "nodes",
    "orchestrator-credits.html": "credits",
    "orchestrator-logs.html": "logs",
    "orchestrator-settings.html": "settings",
    "orchestrator-multipliers.html": "multipliers",
}

HEADER_RE = re.compile(r"<header(?:\s[^>]*)?>.*?</header>", re.DOTALL | re.IGNORECASE)
BODY_RE = re.compile(r"<body(?![^>]*data-active-page)([^>]*)>", re.IGNORECASE)

NODE_HEADER = (
    '<header data-cai-node-header></header>\n'
    '    <script src="/shared/node-header.js"></script>'
)
ORCH_HEADER = (
    '<header data-cai-orch-header></header>\n'
    '    <script src="/shared/orch-header.js"></script>'
)


def patch_node_page(path: Path, active: str) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    if "data-cai-node-header" in text:
        return False
    if not HEADER_RE.search(text):
        return False
    text = HEADER_RE.sub(NODE_HEADER, text, count=1)
    text = BODY_RE.sub(lambda m: f'<body data-active-page="{active}"{m.group(1)}>', text, count=1)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_orch_page(path: Path, active: str) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    if "data-cai-orch-header" in text:
        return False
    if not HEADER_RE.search(text):
        return False
    text = HEADER_RE.sub(ORCH_HEADER, text, count=1)
    text = BODY_RE.sub(lambda m: f'<body data-active-page="{active}"{m.group(1)}>', text, count=1)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    for name, active in NODE_PAGES.items():
        path = NODE_DIR / name
        if path.exists() and patch_node_page(path, active):
            print(f"node: {name}")
    for name, active in ORCH_PAGES.items():
        path = ORCH_DIR / name
        if path.exists() and patch_orch_page(path, active):
            print(f"orch: {name}")


if __name__ == "__main__":
    main()
