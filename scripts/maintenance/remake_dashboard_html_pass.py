"""Batch-remake dashboard HTML: fonts, paraphrased chrome, role=main, keep IDs."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "worker" / "src" / "dashboard" / "static"

FONT_OLD = re.compile(
    r'<link href="https://fonts\.googleapis\.com/css2\?family=Inter[^"]*" rel="stylesheet">\s*'
    r'<link href="https://fonts\.googleapis\.com/css2\?family=Geist\+Mono[^"]*" rel="stylesheet">',
    re.M,
)
FONT_NEW = (
    '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;'
    "0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=IBM+Plex+Mono:wght@400;500;600"
    '&display=swap" rel="stylesheet">'
)

GEIST_FONT = re.compile(r"['\"]Geist Mono['\"],?\s*monospace")
INTER_FONT = re.compile(r"['\"]Inter['\"],?\s*")
GEIST_BARE = re.compile(r"Geist Mono")
INTER_BARE = re.compile(r"\bInter\b")

COPY_SWAPS = [
    ("You are offline. Some features may be unavailable.", "Connection lost. Limited features until you reconnect."),
    ("Orchestrator connection lost.", "Lost contact with the orchestrator."),
    ("View and manage training jobs", "Inspect queued and active training work"),
    ("Manage the DistribAI distributed compute network", "Operate the DistribAI compute mesh"),
    ("Orchestrator Control Panel", "Orchestrator command center"),
    ("Loading nodes...", "Fetching node roster..."),
    ("No recent activity", "Activity log is empty"),
    ("Confirm Action", "Please confirm"),
    ("Are you sure?", "Continue with this action?"),
    ("Session Expiring Soon", "Idle timeout approaching"),
    ("Your session will expire in", "You will be signed out in"),
    ("due to inactivity. Save your work to avoid losing data.", "of idle time. Persist unfinished edits first."),
    ("Stay Active", "Keep session"),
    ("Keyboard Shortcuts", "Hotkeys"),
    ("Activity Feed", "Live activity"),
    ("Clear all", "Reset feed"),
    ("Connected Nodes", "Live nodes"),
    ("Active Nodes", "Online nodes"),
    ("Running Jobs", "Jobs in flight"),
    ("Credits Distributed", "Credits issued"),
    ("TFLOPS Total", "Aggregate TFLOPS"),
]


def patch_html(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text, n = FONT_OLD.subn(FONT_NEW, text)
    if "DM+Sans" not in text and "fonts.googleapis.com" in text:
        text = re.sub(
            r'<link href="https://fonts\.googleapis\.com/css2\?family=[^"]+" rel="stylesheet">\s*',
            FONT_NEW + "\n    ",
            text,
            count=1,
        )
    text = GEIST_FONT.sub("'IBM Plex Mono', monospace", text)
    text = INTER_FONT.sub("'DM Sans', ", text)
    text = GEIST_BARE.sub("IBM Plex Mono", text)
    # Avoid breaking words like "Internal" — only CSS font stacks already handled
    for old, new in COPY_SWAPS:
        text = text.replace(old, new)
    text = re.sub(r"<main(?![^>]*role=)", '<main role="main"', text, count=1)
    text = text.replace("font-family: Inter,", "font-family: 'DM Sans',")
    text = text.replace("font-family:Inter,", "font-family:'DM Sans',")
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    for path in STATIC.rglob("*.html"):
        if patch_html(path):
            changed.append(str(path.relative_to(ROOT)))
    print(f"updated {len(changed)} html files")
    for item in changed:
        print(item)


if __name__ == "__main__":
    main()
