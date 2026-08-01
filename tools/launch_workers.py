"""
launch_workers.py — Spawn N worker daemons on this machine for testing.
Each worker gets a unique node ID derived from hostname + index.
Ctrl+C kills all workers.
Usage:
  python tools/launch_workers.py
  python tools/launch_workers.py --count 5
  python tools/launch_workers.py --count 3 --orchestrator localhost:50051
  python tools/launch_workers.py --count 3 --log-level DEBUG
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch multiple worker daemons")
    parser.add_argument("--count", "-n", type=int, default=3, help="Number of workers (default: 3)")
    parser.add_argument(
        "--orchestrator",
        default="localhost:50051",
        help="Orchestrator gRPC address",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--state-dir",
        default=str(ROOT / "worker" / "runtime"),
        help="State directory (shared across workers)",
    )
    args = parser.parse_args()
    python = sys.executable
    procs = []
    print(f"\nLaunching {args.count} worker(s) → {args.orchestrator}")
    print(f"State dir : {args.state_dir}")
    print(f"Log level : {args.log_level}\n")
    for i in range(1, args.count + 1):
        cmd = [
            python,
            "-m",
            "worker.src.daemon.run",
            "--orchestrator",
            args.orchestrator,
            "--state-dir",
            args.state_dir,
            "--log-level",
            args.log_level,
            "--worker-index",
            str(i),
        ]
        env = {**os.environ, "WORKER_INDEX": str(i)}
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
        procs.append((i, proc))
        print(f"  Worker {i:02d} pid={proc.pid}")
        time.sleep(0.25)
    print(f"\n{args.count} worker(s) running. Ctrl+C to stop all.\n")

    def _shutdown(sig, frame) -> None:
        print("\nStopping all workers…")
        for _, p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(2)
        for _, p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except ProcessLookupError:
                    pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    for _, p in procs:
        p.wait()


if __name__ == "__main__":
    main()
