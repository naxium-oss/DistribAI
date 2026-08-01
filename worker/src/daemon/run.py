"""
Worker daemon CLI: ``python -m worker.src.daemon.run``.

Examples:
  python -m worker.src.daemon.run
  python -m worker.src.daemon.run --orchestrator host:50051
  python -m worker.src.daemon.run --node-id my-gpu-box --log-level DEBUG
"""

import argparse
import asyncio
import logging
import os
import secrets
import socket
import sys


def _setup_logging(level: str, node_id: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    if level.upper() != "DEBUG":
        logging.getLogger("aiohttp").setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DistribAI Worker Daemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m worker.src.daemon.run
  python -m worker.src.daemon.run --node-id my-rtx4090 --log-level DEBUG
  python -m worker.src.daemon.run --orchestrator remote-host:50051
        """,
    )
    parser.add_argument(
        "--orchestrator",
        "-o",
        default=os.environ.get("ORCHESTRATOR_URL", "localhost:50051"),
        help="Orchestrator gRPC address (default: localhost:50051)",
    )
    parser.add_argument(
        "--node-id",
        "-n",
        default=os.environ.get("NODE_ID"),
        help="Node ID (auto-generated from hostname if omitted)",
    )
    parser.add_argument(
        "--state-dir",
        "-s",
        default=os.environ.get("STATE_DIR"),
        help="Directory for state files (default: worker/runtime/)",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--worker-index",
        "-i",
        type=int,
        default=int(os.environ.get("WORKER_INDEX", "0")),
        help="Worker index for multi-worker testing (affects auto node-id)",
    )
    args = parser.parse_args()
    node_id = args.node_id
    if not node_id:
        host = socket.gethostname().split(".")[0]
        suffix = secrets.token_hex(8)
        node_id = (
            f"{host}-w{args.worker_index:02d}-{suffix}"
            if args.worker_index > 0
            else f"{host}-{suffix}"
        )
    _setup_logging(args.log_level, node_id)
    from .daemon import WorkerDaemon

    daemon = WorkerDaemon(
        orchestrator_url=args.orchestrator,
        node_id=node_id,
        state_dir=args.state_dir,
        worker_index=args.worker_index,
    )
    print(f"\n{'─' * 56}")
    print("  DistribAI Worker Daemon")
    print(f"  Node ID   : {node_id}")
    print(f"  Orchestrator: {args.orchestrator}")
    print(f"{'─' * 56}\n")
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print(f"\n[{node_id}] Interrupted — shutting down")


if __name__ == "__main__":
    main()
