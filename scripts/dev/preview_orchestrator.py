"""Run the DistribAI orchestrator locally without packaging.

This starts the orchestrator service directly with visible console output,
so you can monitor its operations, logs, and debug in real-time.
No installer, PyInstaller bundle, or packaging required.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _wait_ready(url: str, timeout: float) -> bool:
    """Wait for the admin API to become ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status < 500
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.25)
    return False


def find_python() -> str:
    """Find the appropriate Python executable."""
    candidates = [
        "C:\\Program Files\\Python311\\python.exe",
        "C:\\Program Files\\Python310\\python.exe",
        "C:\\Program Files\\Python312\\python.exe",
        "python",
        "python3",
        "py",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    import shutil

    py = shutil.which("python") or shutil.which("python3") or shutil.which("py")
    if py:
        return py

    return sys.executable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the DistribAI orchestrator locally")
    parser.add_argument(
        "--grpc-port", type=int, default=50051, help="gRPC service port (default: 50051)"
    )
    parser.add_argument(
        "--admin-port", type=int, default=8766, help="Admin API port (default: 8766)"
    )
    parser.add_argument(
        "--admin-host", type=str, default="127.0.0.1", help="Admin API host (default: 127.0.0.1)"
    )
    parser.add_argument("--no-browser", action="store_true", help="Start without opening a browser")
    parser.add_argument(
        "--no-gui", action="store_true", help="Run without the GUI window (console only)"
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Startup timeout in seconds")
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable path (auto-detected if not specified)",
    )
    args = parser.parse_args(argv)

    python = args.python or find_python()

    orchestrator_module = REPO_ROOT / "services_python" / "orchestrator_grpc.py"
    if not orchestrator_module.exists():
        sys.stderr.write(f"Missing orchestrator module: {orchestrator_module}\n")
        return 1

    try:
        import_result = subprocess.run(
            [python, "-c", "import grpc, aiohttp, boto3, jwt"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if import_result.returncode != 0:
            sys.stderr.write("Missing dependencies. Install with:\n")
            sys.stderr.write(
                f"  {python} -m pip install grpcio aiohttp boto3 pyjwt python-dotenv aiohttp_cors\n"
            )
            return 1
    except Exception as e:
        sys.stderr.write(f"Error checking dependencies: {e}\n")
        return 1

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "GRPC_PORT": str(args.grpc_port),
        "ADMIN_PORT": str(args.admin_port),
        "ADMIN_HOST": args.admin_host,
    }

    creationflags = 0
    if args.no_gui and sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW

    print("Starting DistribAI Orchestrator...")
    print(f"  gRPC port: {args.grpc_port}")
    print(f"  Admin API: http://{args.admin_host}:{args.admin_port}")
    print(f"  Python: {python}")
    print("")

    cmd = [python, "-m", "services_python.orchestrator_grpc"]

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            creationflags=creationflags,
        )

        admin_url = f"http://{args.admin_host}:{args.admin_port}/"
        health_url = f"{admin_url}admin/health"

        if not _wait_ready(health_url, args.timeout):
            sys.stderr.write("Orchestrator did not become ready in time.\n")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 1

        print("Orchestrator is running!")
        print(f"  gRPC endpoint: localhost:{args.grpc_port}")
        print(f"  Admin API: {admin_url}")
        print(f"  Health check: {health_url}")
        print("")
        print("Press Ctrl+C to stop.")
        print("")

        if not args.no_browser:
            webbrowser.open(admin_url)

        while proc.poll() is None:
            time.sleep(0.5)

        return proc.returncode or 0

    except KeyboardInterrupt:
        print("\nStopping orchestrator...")
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
