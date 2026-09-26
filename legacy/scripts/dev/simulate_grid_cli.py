#!/usr/bin/env python3
"""
Subprocess-based grid simulator CLI (multi-worker + orchestrator on one machine).

Prefer this for quick process-level bring-up. For the in-process threaded harness
(real orchestrator/worker in threads), see ``tools/simulate_grid.py``.

Run from repository root:

    python -m scripts.dev.simulate_grid_cli --workers 3
"""

import argparse
import signal
import subprocess
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Simulate a complete DistribAI system (subprocess orchestrator + workers)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from repo root):
  python -m scripts.dev.simulate_grid_cli --workers 3
  python -m scripts.dev.simulate_grid_cli --workers 5 --grpc-port 20001
  python -m scripts.dev.simulate_grid_cli --workers 2 --admin-port 19002
        """,
    )

    parser.add_argument(
        "--workers", type=int, default=3, help="Number of worker processes to start (default: 3)"
    )

    parser.add_argument(
        "--grpc-port",
        type=int,
        default=19001,
        help="gRPC port for the orchestrator (default: 19001)",
    )

    parser.add_argument(
        "--admin-port",
        type=int,
        default=19002,
        help="Admin API port for the orchestrator (default: 19002)",
    )

    parser.add_argument(
        "--worker-port-start",
        type=int,
        default=19100,
        help="Starting port for worker gRPC connections (default: 19100)",
    )

    parser.add_argument(
        "--timeout", type=int, default=30, help="Timeout in seconds for startup (default: 30)"
    )

    parser.add_argument(
        "--no-orchestrator",
        action="store_true",
        help="Skip starting the orchestrator (use existing one)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


class GridSimulator:
    """Manages the simulated grid environment."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.orchestrator_proc: subprocess.Popen | None = None
        self.worker_procs: list[subprocess.Popen] = []
        self.running = True

    def start_orchestrator(self) -> None:
        """Start the orchestrator process."""
        cmd = [
            sys.executable,
            "-m",
            "services_python.orchestrator_grpc",
            "--grpc-port",
            str(self.args.grpc_port),
            "--admin-port",
            str(self.args.admin_port),
        ]

        if self.args.verbose:
            cmd.append("--verbose")

        print(f"Starting orchestrator: {' '.join(cmd)}")
        self.orchestrator_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not self.args.verbose else None,
            stderr=subprocess.PIPE if not self.args.verbose else None,
            text=True,
        )

    def start_workers(self) -> None:
        """Start worker processes."""
        for i in range(self.args.workers):
            worker_port = self.args.worker_port_start + i
            cmd = [
                sys.executable,
                "-m",
                "worker.src.daemon.run",
                "--orchestrator-address",
                f"localhost:{self.args.grpc_port}",
                "--worker-port",
                str(worker_port),
            ]

            if self.args.verbose:
                cmd.append("--verbose")

            print(f"Starting worker {i + 1}: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE if not self.args.verbose else None,
                stderr=subprocess.PIPE if not self.args.verbose else None,
                text=True,
            )
            self.worker_procs.append(proc)

    def wait_for_startup(self) -> None:
        """Wait for all processes to start up."""
        print(f"Waiting {self.args.timeout}s for startup...")
        time.sleep(self.args.timeout)

        if self.orchestrator_proc and self.orchestrator_proc.poll() is not None:
            raise RuntimeError("Orchestrator failed to start")

        failed_workers = []
        for i, proc in enumerate(self.worker_procs):
            if proc.poll() is not None:
                failed_workers.append(i + 1)

        if failed_workers:
            raise RuntimeError(f"Workers {failed_workers} failed to start")

        print("All processes started successfully!")

    def stop_all(self) -> None:
        """Stop all processes."""
        print("Stopping all processes...")
        self.running = False

        for proc in self.worker_procs:
            if proc.poll() is None:
                proc.terminate()

        if self.orchestrator_proc and self.orchestrator_proc.poll() is None:
            self.orchestrator_proc.terminate()

        time.sleep(2)

        for proc in self.worker_procs:
            if proc.poll() is None:
                proc.kill()

        if self.orchestrator_proc and self.orchestrator_proc.poll() is None:
            self.orchestrator_proc.kill()

    def run(self) -> None:
        """Run the simulation."""
        try:
            if not self.args.no_orchestrator:
                self.start_orchestrator()

            self.start_workers()
            self.wait_for_startup()

            print(f"Simulation running with {self.args.workers} workers")
            print(f"Orchestrator: gRPC={self.args.grpc_port}, admin={self.args.admin_port}")
            print("Press Ctrl+C to stop...")

            while self.running:
                time.sleep(1)

        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
        except Exception as e:
            print(f"Simulation failed: {e}")
            sys.exit(1)
        finally:
            self.stop_all()


def signal_handler(signum: int, frame) -> None:
    """Handle interrupt signals."""
    print(f"\nReceived signal {signum}")


def main() -> None:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    simulator = GridSimulator(args)
    simulator.run()


if __name__ == "__main__":
    main()
