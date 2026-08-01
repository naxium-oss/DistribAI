"""
Multi-device / multi-architecture training verification harness.

Boots one real orchestrator plus several real ``WorkerDaemon`` instances
in-process (each with its own node id, state directory, and a capped
resource-limit profile — standing in for separate physical machines or VMs),
then submits a small real training job for every registered architecture
family plus a GQA/qk-norm/head-gating decoder variant. Each job must reach
"success" before the harness reports pass/fail per architecture.

Run from the repository root:

    python -m tools.verify_multi_device_training
    python -m tools.verify_multi_device_training --workers 5 --cpu-percent 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from services_python.orchestrator_grpc import serve as orchestrator_serve
from worker.src.daemon.daemon import WorkerDaemon

ARCHITECTURE_MATRIX: list[dict] = [
    {"family": "decoder_transformer", "dim": 32, "n_unique_layers": 2, "n_logical_layers": 2, "n_heads": 4, "ffn_dim": 64, "seq_len": 32},
    {"family": "decoder_transformer", "dim": 32, "n_unique_layers": 2, "n_logical_layers": 4, "n_heads": 8, "n_kv_heads": 2, "ffn_dim": 64, "seq_len": 32, "qk_norm": True, "use_head_gating": True, "embedding_scale": True, "attn_res_block_size": 2},
    {"family": "gru", "dim": 32, "gru_layers": 2, "seq_len": 32},
    {"family": "lstm", "dim": 32, "gru_layers": 2, "seq_len": 32},
    {"family": "gated_conv", "dim": 32, "n_logical_layers": 2, "conv_kernel": 3, "seq_len": 32},
    {"family": "resnet_lm", "dim": 32, "n_logical_layers": 2, "conv_kernel": 3, "seq_len": 32},
    {"family": "moe_decoder", "dim": 32, "ffn_dim": 64, "n_logical_layers": 2, "num_experts": 4, "top_k": 2, "seq_len": 32},
    {"family": "hybrid_attn_rnn", "dim": 32, "n_logical_layers": 2, "n_heads": 4, "ffn_dim": 64, "seq_len": 32},
    {"family": "dense_ffn", "dim": 32, "ffn_dim": 64, "n_logical_layers": 2, "seq_len": 32},
]


def _pick_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _configure_harness_env(grpc_port: int, admin_port: int) -> None:
    os.environ["GRPC_PORT"] = str(grpc_port)
    os.environ["ADMIN_PORT"] = str(admin_port)
    os.environ["DISTRIBAI_API_URL"] = f"http://127.0.0.1:{admin_port}"
    os.environ.setdefault("ADMIN_REQUIRE_AUTH", "")
    os.environ.setdefault("ADMIN_HOST", "127.0.0.1")
    os.environ.setdefault("RATE_LIMIT_DISABLED", "1")


class OrchestratorThread:
    def __init__(self, grpc_port: int, admin_port: int) -> None:
        self.grpc_port = grpc_port
        self.admin_port = admin_port
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.start_error: BaseException | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.start_error is not None:
                raise RuntimeError(f"orchestrator failed to start: {self.start_error}") from self.start_error
            if self.thread and not self.thread.is_alive():
                raise RuntimeError("orchestrator thread exited before binding ports")
            if self.loop is not None:
                return
            time.sleep(0.05)
        raise RuntimeError("orchestrator thread did not initialize its event loop in time")

    def stop(self) -> None:
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        _configure_harness_env(self.grpc_port, self.admin_port)

        async def run_orch() -> None:
            await orchestrator_serve(block=True)

        try:
            self.loop.run_until_complete(run_orch())
        except RuntimeError as exc:
            if "Event loop stopped before Future completed" not in str(exc):
                self.start_error = exc
                raise
        except BaseException as exc:
            self.start_error = exc
            raise


def _write_resource_cap(home_dir: str, cpu_percent: int, gpu_percent: int, ram_percent: int) -> None:
    config_dir = os.path.join(home_dir, ".distribai")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "desktop.json"), "w", encoding="utf-8") as fh:
        json.dump({"cpuPercent": cpu_percent, "gpuPercent": gpu_percent, "ramPercent": ram_percent}, fh)


def _seed_benchmark_results(node_id: str) -> None:
    """Skip the ~60s benchmark subprocess during harness runs."""
    results_file = Path(tempfile.gettempdir()) / f"distribai_benchmark_results_{node_id}.json"
    payload = {
        "type": "suite_complete",
        "overall_score": 50.0,
        "tier": "standard",
        "harness_seeded": True,
    }
    results_file.write_text(json.dumps(payload), encoding="utf-8")


def _ensure_sample_batch_path() -> str:
    """Small real text batch under runtime/ (the only allowlisted local blob root)."""
    repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    batch_dir = os.path.join(repo_root, "runtime", "verify_batches")
    os.makedirs(batch_dir, exist_ok=True)
    batch_path = os.path.join(batch_dir, "sample.txt")
    if not os.path.isfile(batch_path):
        with open(batch_path, "w", encoding="utf-8") as fh:
            fh.write(
                "The quick brown fox jumps over the lazy dog. Distributed training "
                "across many devices exercises every registered architecture family "
                "with small synthetic batches so the whole pipeline stays honest end to end."
            )
    return batch_path


async def _wait_for_admin(admin_port: int, timeout_s: float = 30.0) -> None:
    url = f"http://127.0.0.1:{admin_port}/admin/health"
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return
            except (TimeoutError, aiohttp.ClientError):
                pass
            await asyncio.sleep(0.25)
    raise RuntimeError(f"admin API not ready at {url}")


async def _wait_for_nodes(admin_port: int, expected: int, timeout_s: float = 90.0) -> None:
    url = f"http://127.0.0.1:{admin_port}/admin/nodes"
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    body = await resp.json()
                nodes = body.get("nodes") or body.get("active_nodes") or []
                if len(nodes) >= expected:
                    return
            except (TimeoutError, aiohttp.ClientError, json.JSONDecodeError):
                pass
            await asyncio.sleep(0.5)
    raise RuntimeError(f"expected {expected} registered nodes, timed out waiting on {url}")


async def _submit_job(admin_port: int, architecture_config: dict) -> str:
    async with aiohttp.ClientSession() as session:
        payload = {
            "model_name": "uploaded-architecture",
            "base_model": "uploaded-architecture",
            "architecture_config": architecture_config,
            "batch_blob_url": _ensure_sample_batch_path(),
            "steps": 1,
            "batch_size": 4,
            "job_type": "fine_tune",
        }
        async with session.post(f"http://127.0.0.1:{admin_port}/admin/jobs", json=payload) as resp:
            body = await resp.json()
            if not body.get("ok", True) and "job_id" not in body:
                raise RuntimeError(f"job creation failed: {body}")
            return body["job_id"]


async def _await_job_status(admin_port: int, job_id: str, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            async with session.get(f"http://127.0.0.1:{admin_port}/admin/jobs/{job_id}") as resp:
                body = await resp.json()
            status = body.get("status") or body.get("job", {}).get("status")
            if status in {"success", "completed", "failed", "cancelled"}:
                return status
            await asyncio.sleep(1.0)
    return "timeout"


async def run_verification(
    num_workers: int, cpu_percent: int, gpu_percent: int, ram_percent: int, grpc_port: int, admin_port: int
) -> int:
    if grpc_port <= 0:
        grpc_port = _pick_free_port()
    if admin_port <= 0:
        admin_port = _pick_free_port()

    _configure_harness_env(grpc_port, admin_port)
    print(f"Using gRPC port {grpc_port}, admin port {admin_port}")

    orch = OrchestratorThread(grpc_port, admin_port)
    orch.start()
    await _wait_for_admin(admin_port)

    device_homes = [tempfile.mkdtemp(prefix=f"distribai-device{i}-") for i in range(num_workers)]
    for home_dir in device_homes:
        _write_resource_cap(home_dir, cpu_percent, gpu_percent, ram_percent)

    real_home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    workers: list[WorkerDaemon] = []
    tasks: list[asyncio.Task] = []
    try:
        for index, home_dir in enumerate(device_homes):
            os.environ["USERPROFILE"] = home_dir
            os.environ["HOME"] = home_dir
            state_dir = tempfile.mkdtemp(prefix=f"distribai-state{index}-")
            worker = WorkerDaemon(
                orchestrator_url=f"127.0.0.1:{grpc_port}",
                worker_index=index + 1,
                state_dir=state_dir,
            )
            _seed_benchmark_results(worker.node_id)
            workers.append(worker)
            tasks.append(asyncio.create_task(worker.run()))

        if real_home:
            os.environ["HOME"] = real_home
            os.environ["USERPROFILE"] = real_home

        node_ids = {w.node_id for w in workers}
        print(f"Simulated devices: {len(workers)} (unique node ids: {len(node_ids)})")
        assert len(node_ids) == len(workers), "each simulated device must get a distinct node id"

        await _wait_for_nodes(admin_port, num_workers)

        results: dict[str, str] = {}
        for spec in ARCHITECTURE_MATRIX:
            label = spec["family"] + ("+gqa" if spec.get("n_kv_heads") else "")
            try:
                job_id = await _submit_job(admin_port, spec)
                status = await _await_job_status(admin_port, job_id, timeout_s=120)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                status = f"error: {exc}"
            results[label] = status
            print(f"[{label}] -> {status}")

        failures = [label for label, status in results.items() if status not in {"success", "completed"}]
        print("\n=== Summary ===")
        for label, status in results.items():
            print(f"  {label}: {status}")
        if failures:
            print(f"\nFAILED architectures: {failures}")
            return 1
        print("\nAll architectures trained successfully across simulated devices.")
        return 0
    finally:
        for worker in workers:
            worker.stop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        orch.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4, help="Number of simulated devices/VMs")
    parser.add_argument("--cpu-percent", type=int, default=30, help="Per-device CPU cap")
    parser.add_argument("--gpu-percent", type=int, default=30, help="Per-device GPU cap")
    parser.add_argument("--ram-percent", type=int, default=30, help="Per-device RAM cap")
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=0,
        help="gRPC port (0 = pick a free port automatically)",
    )
    parser.add_argument(
        "--admin-port",
        type=int,
        default=0,
        help="Admin HTTP port (0 = pick a free port automatically)",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(
        run_verification(
            args.workers,
            args.cpu_percent,
            args.gpu_percent,
            args.ram_percent,
            args.grpc_port,
            args.admin_port,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
