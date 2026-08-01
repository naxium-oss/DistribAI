#!/usr/bin/env python3
"""Rewrite the public-grid host baked into the Node EXE runtime hook.

This keeps the source of truth in `runtime/secrets/release/public_host.txt`
(or `PUBLIC_GRID_HOST` env var) and updates the constants in
`worker/src/daemon/_node_defaults.py` before the next PyInstaller build.

Usage:
    PUBLIC_GRID_HOST=grid.example.com python scripts/packaging/inject_public_host.py
    python scripts/packaging/inject_public_host.py --host grid.example.com --port 50051
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / "worker" / "src" / "daemon" / "_node_defaults.py"
HOST_FILE = REPO_ROOT / "runtime" / "secrets" / "release" / "public_host.txt"

HOST_RE = re.compile(r'PUBLIC_GRID_HOST:\s*str\s*=\s*"([^"]+)"')
PORT_RE = re.compile(r'PUBLIC_GRID_PORT:\s*str\s*=\s*"([^"]+)"')


def resolve_host(args: argparse.Namespace) -> tuple[str, str]:
    host = args.host or ""
    if not host:
        host = HOST_FILE.read_text(encoding="utf-8").strip() if HOST_FILE.is_file() else ""
    if not host:
        host = "127.0.0.1"
    return host, args.port


def rewrite_hook(host: str, port: str) -> None:
    text = HOOK_PATH.read_text(encoding="utf-8")
    if HOST_RE.search(text):
        text = HOST_RE.sub(f'PUBLIC_GRID_HOST: str = "{host}"', text, count=1)
    else:
        text = text.replace(
            'PUBLIC_GRID_HOST: str = "127.0.0.1"',
            f'PUBLIC_GRID_HOST: str = "{host}"',
            1,
        )
    if PORT_RE.search(text):
        text = PORT_RE.sub(f'PUBLIC_GRID_PORT: str = "{port}"', text, count=1)
    else:
        text = text.replace(
            'PUBLIC_GRID_PORT: str = "50051"',
            f'PUBLIC_GRID_PORT: str = "{port}"',
            1,
        )
    HOOK_PATH.write_text(text, encoding="utf-8")
    print(f"[inject_public_host] {HOOK_PATH} -> {host}:{port}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", help="Public host/IP for the orchestrator gRPC endpoint")
    p.add_argument("--port", default="50051", help="gRPC port (default: 50051)")
    args = p.parse_args(argv)
    host, port = resolve_host(args)
    rewrite_hook(host, port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
