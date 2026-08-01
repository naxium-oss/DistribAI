"""Open the DistribAI dashboard locally without packaging.

This starts the existing Node dashboard shell and the local orchestrator when requested.
No installer or PyInstaller bundle is required; all displayed data comes from live local services.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _wait_ready(url: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status < 500
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.25)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview the DistribAI dashboard locally")
    parser.add_argument("--port", type=int, default=3210, help="Local HTTP port")
    parser.add_argument("--role", choices=["node", "admin"], default="admin", help="Dashboard role")
    parser.add_argument(
        "--no-browser", action="store_true", help="Start server without opening a browser"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Startup timeout in seconds")
    args = parser.parse_args(argv)

    node = shutil.which("node") or shutil.which("node.exe")
    if not node:
        sys.stderr.write(
            "Node.js is required for dashboard preview. Install Node.js, then run npm install.\n"
        )
        return 1

    server_js = REPO_ROOT / "client" / "server.js"
    if not server_js.exists():
        sys.stderr.write(f"Missing dashboard server: {server_js}\n")
        return 1

    env = {
        **os.environ,
        "PORT": str(args.port),
        "AUTO_START_ORCH": "1",
        "PYTHONUNBUFFERED": "1",
    }
    proc = subprocess.Popen([node, str(server_js)], cwd=str(REPO_ROOT), env=env)
    url = f"http://127.0.0.1:{args.port}/?role={args.role}&preview=1"
    try:
        if not _wait_ready(f"http://127.0.0.1:{args.port}/", args.timeout):
            sys.stderr.write("Dashboard preview did not become ready in time.\n")
            return 1
        print(f"Dashboard preview: {url}")
        print("Press Ctrl+C to stop.")
        if not args.no_browser:
            webbrowser.open(url)
        while proc.poll() is None:
            time.sleep(0.5)
        return proc.returncode or 0
    except KeyboardInterrupt:
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
