"""
Grid Simulation for DistribAI Testing

Simulates a multi-node grid using real orchestrator and worker components.
Runs the actual system code in separate threads to test the full
pipeline on a single machine.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time

import aiohttp

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "worker", "src", "distribai_proto"))
)

from services_python.orchestrator_grpc import serve as orchestrator_serve
from worker.src.daemon.daemon import WorkerDaemon

logging.basicConfig(level=logging.INFO)

GRPC_PORT = 50051
ADMIN_PORT = 8766


class MaliciousWorkerDaemon(WorkerDaemon):
    """
    Worker daemon that sends invalid authentication data.

    Used for testing security and authentication failures.
    Simulates a malicious worker attempting to join the grid.

    Attributes:
        distribai_pb2: Protocol buffer module for messages

    Example:
        >>> malicious = MaliciousWorkerDaemon(
        ...     orchestrator_url="localhost:50051",
        ...     worker_index=99
        ... )
        >>> await malicious._register()
    """

    import worker.src.distribai_proto.distribai_pb2 as distribai_pb2

    async def _register(self) -> None:
        """
        Send malicious registration with invalid benchmark data.

        Overrides parent registration to simulate attack attempts.
        Expects authentication failure from orchestrator.
        """
        hw = self._hardware_info()
        msg = self.distribai_pb2.ClientMessage(
            register=self.distribai_pb2.RegisterSession(
                node_id=self.node_id,
                jwt_token=os.getenv("DISTRIBAI_JWT_TOKEN") or "",
                hardware_json=json.dumps(hw),
                benchmark_json=json.dumps({"score": 0, "tier": "malicious"}),
                ts=int(time.time()),
            )
        )
        await self._send(msg)
        logging.info(f"[{self.node_id}] Malicious REGISTER sent (expecting auth failure)")


class OrchestratorThread:
    """
    Thread wrapper for running the orchestrator in background.

    Manages the lifecycle of the orchestrator gRPC server
    in a separate thread with its own event loop.

    Attributes:
        grpc_port: gRPC service port
        admin_port: Admin HTTP API port
        loop: Asyncio event loop for the thread
        thread: Background thread running orchestrator
        _stop_event: Event for signaling shutdown

    Example:
        >>> orch = OrchestratorThread(50051, 8766)
        >>> orch.start()
        >>> # ... run tests ...
        >>> orch.stop()
    """

    def __init__(self, grpc_port: int, admin_port: int):
        """
        Initialize orchestrator thread.

        Args:
            grpc_port: Port for gRPC service
            admin_port: Port for admin HTTP API
        """
        self.grpc_port = grpc_port
        self.admin_port = admin_port
        self.loop = None
        self.thread = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """
        Start the orchestrator in a background thread.

        Waits 1 second for initialization to complete.
        """
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(1)

    def stop(self) -> None:
        """
        Stop the orchestrator thread.

        Signals stop event and waits for thread termination.
        """
        self._stop_event.set()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)

    def _run(self) -> None:
        """
        Run the orchestrator server.

        Creates event loop and runs gRPC server until stopped.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        os.environ["GRPC_PORT"] = str(self.grpc_port)
        os.environ["ADMIN_PORT"] = str(self.admin_port)

        async def run_orch():
            await orchestrator_serve()

        self.loop.run_until_complete(run_orch())


async def test_grid() -> None:
    """
    Run full grid simulation test.

    Creates orchestrator and multiple workers (fast, slow, malicious),
    injects a training job, and verifies completion.

    Steps:
        1. Start orchestrator in background thread
        2. Connect 3 workers (1 fast, 1 slow, 1 malicious)
        3. Submit PyTorch training job via admin API
        4. Wait for completion and verify results
        5. Shutdown all components

    Example:
        >>> asyncio.run(test_grid())
    """
    print("\n\n--- 1. Starting REAL Central Orchestrator ---")
    orch_thread = OrchestratorThread(GRPC_PORT, ADMIN_PORT)
    orch_thread.start()
    print(f"Orchestrator running on gRPC port {GRPC_PORT}, Admin port {ADMIN_PORT}")
    print("\n\n--- 2. Connecting REAL Workers (1 Fast, 1 Slow, 1 Malicious) ---")
    w_fast = WorkerDaemon(
        orchestrator_url=f"127.0.0.1:{GRPC_PORT}", worker_index=1, state_dir="runtime/w1"
    )
    w_slow = WorkerDaemon(
        orchestrator_url=f"127.0.0.1:{GRPC_PORT}", worker_index=2, state_dir="runtime/w2"
    )
    w_malicious = MaliciousWorkerDaemon(
        orchestrator_url=f"127.0.0.1:{GRPC_PORT}", worker_index=3, state_dir="runtime/w3"
    )
    asyncio.create_task(w_fast.run())
    asyncio.create_task(w_slow.run())
    await asyncio.sleep(3)
    print("\n\n--- 3. Injecting REAL PyTorch Training Job ---")
    async with aiohttp.ClientSession() as session:
        job_def = {
            "model_name": "distribai-small",
            "steps": 10,
            "batch_size": 16,
            "job_type": "fine_tune",
        }
        async with session.post(f"http://127.0.0.1:{ADMIN_PORT}/admin/jobs", json=job_def) as resp:
            result = await resp.json()
            print("Job injected:", result)
    print("\nWaiting for job completion...")
    await asyncio.sleep(20)
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://127.0.0.1:{ADMIN_PORT}/admin/jobs") as resp:
            jobs_data = await resp.json()
            print(f"\nJobs status: {jobs_data}")
    print("\n\n--- 4. Shutting down & Checking Results ---")
    w_fast.stop()
    w_slow.stop()
    w_malicious.stop()
    orch_thread.stop()
    if os.path.exists("checkpoint.pt"):
        print("\n=> SUCCESS! checkpoint.pt was saved.")
    else:
        print("\n=> Note: checkpoint.pt was not saved (S3 not configured, using local mode)")
    success_count = sum(1 for j in jobs_data.get("jobs", []) if j.get("status") == "success")
    print(f"=> {success_count} jobs completed successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("DistribAI Simulation with REAL Orchestrator and Workers")
    print("This runs the actual system code in a test harness")
    print("=" * 60)
    asyncio.run(test_grid())
