"""
Worker Daemon for DistribAI

Manages the worker node lifecycle, including task execution, heartbeat management,
and communication with the orchestrator via gRPC.
"""

import asyncio
import base64
import json
import logging
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from typing import Any

import grpc
import psutil
import torch

try:
    from worker.src.distribai_proto import distribai_pb2, distribai_pb2_grpc
except ImportError:
    from distribai_proto import distribai_pb2, distribai_pb2_grpc
from .bench_manager import BenchmarkManager
from .registration import RegistrationManager
from .state import WorkerState

logger = logging.getLogger(__name__)


def _sanitize_log_message(msg: str) -> str:
    """Sanitize log messages by redacting sensitive information."""
    if not msg:
        return ""
    jwt_pattern = r"ey[a-zA-Z0-9_-]+\.ey[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"
    msg = re.sub(jwt_pattern, "[REDACTED_JWT]", msg)
    msg = re.sub(r"(Authorization:?\s*)(\S+)", r"\1[REDACTED]", msg, flags=re.IGNORECASE)
    msg = re.sub(
        r"(password|secret|token|key)(\s*[:=]\s*)(\S+)", r"\1\2[REDACTED]", msg, flags=re.IGNORECASE
    )
    sanitized = "".join(c for c in msg if c.isprintable() or c in "\n\r\t")
    return sanitized[:1000] if len(sanitized) > 1000 else sanitized


class RedactingFilter(logging.Filter):
    """Logging filter that redacts sensitive information."""

    def filter(self, record):
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        record.msg = _sanitize_log_message(message)
        record.args = ()
        return True


logger.addFilter(RedactingFilter())


def _get_jwt_token() -> str | None:
    """Get JWT token from environment."""
    return os.getenv("DISTRIBAI_JWT_TOKEN")


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


PROGRESS_THROTTLE_SECONDS = 2.0
PROGRESS_MILESTONE_STEPS = 10


class WorkerDaemon:
    """
    Main worker daemon for DistribAI.

    Manages the worker node lifecycle including connection to the orchestrator,
    task execution, heartbeat management, and automatic reconnection.

    Attributes:
        orchestrator_url: URL of the gRPC orchestrator
        node_id: Unique identifier for this worker node
        worker_index: Index for multiple workers on the same machine
        state: WorkerState instance for persistent state management
        executor: JobExecutor for running training tasks
        bench: BenchmarkManager for hardware benchmarking

    Example:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="worker-001"
        )
        await daemon.run()
    """

    HEARTBEAT_INTERVAL = 10
    RECONNECT_DELAY = 5
    MAX_RECONNECT_DELAY = 60

    def __init__(
        self,
        orchestrator_url: str,
        node_id: str | None = None,
        state_dir: str | None = None,
        worker_index: int = 0,
    ) -> None:
        """
        Initialize the worker daemon.

        Args:
            orchestrator_url: URL of the gRPC orchestrator (e.g., "localhost:50051")
            node_id: Optional node ID (auto-generated if not provided)
            state_dir: Directory for persistent state storage
            worker_index: Index for multiple workers on same machine

        Example:
            >>> daemon = WorkerDaemon(
            ...     orchestrator_url="localhost:50051",
            ...     node_id="worker-001",
            ...     worker_index=0
            ... )
        """
        self.orchestrator_url = (
            orchestrator_url.replace("http://", "")
            .replace("https://", "")
            .replace("ws://", "")
        )
        self.node_id = node_id or self._make_node_id(worker_index)
        self.worker_index = worker_index
        self._seq = 0
        self._session_token: str | None = None
        self._current_job: dict[str, Any] | None = None
        self._execution_task: asyncio.Task | None = None
        self._benchmark_task: asyncio.Task | None = None
        self._benchmark_requested = False
        self._grpo_runner = None  # lazy: GrpoRunner instance for GRPO jobs
        self._grpo_round_task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self.connected = False
        self._send_queue: asyncio.Queue | None = None
        self._ephemeral = _env_flag("DISTRIBAI_EPHEMERAL")
        if self._ephemeral and not state_dir:
            state_dir = tempfile.mkdtemp(prefix="distribai-ephemeral-")
            logger.info("[%s] Ephemeral mode — state under %s", self.node_id, state_dir)
        state_root = state_dir or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "runtime"
        )
        self.state = WorkerState(
            state_dir=state_root, node_id=self.node_id, ephemeral=self._ephemeral
        )
        auth = self.state.load_auth_tokens()
        self._session_token = auth.session_token
        self.executor = None
        self._script_runner = None
        self.bench = BenchmarkManager(node_id=self.node_id)
        self._last_progress_report = 0

    def _get_executor(self):
        if self.executor is None:
            from .executor import JobExecutor

            self.executor = JobExecutor(
                node_id=self.node_id,
                on_progress=self._report_progress,
                on_result=self._report_result,
            )
        return self.executor

    async def run(self) -> None:
        """
        Run the worker daemon main loop.

        Continuously connects to the orchestrator, processes tasks,
        and handles automatic reconnection on failure.

        Example:
            >>> daemon = WorkerDaemon("localhost:50051")
            >>> await daemon.run()
        """
        delay = self.RECONNECT_DELAY
        self.state.set_status("starting")
        logger.info("[%s] Starting daemon → %s", self.node_id, self.orchestrator_url)
        while not self._stop.is_set():
            try:
                await self._connect_and_run()
                delay = self.RECONNECT_DELAY
            except grpc.aio.AioRpcError as e:
                logger.warning("[%s] RPC error: %s %s", self.node_id, e.code(), e.details())
                if e.code() == grpc.StatusCode.UNAUTHENTICATED:
                    self._session_token = None
                    self.state.clear_auth_tokens()
            except Exception as e:
                logger.error("[%s] Unexpected: %s", self.node_id, e, exc_info=True)
            if not self._stop.is_set():
                logger.info("[%s] Reconnecting in %ss…", self.node_id, delay)
                self.state.set_status("reconnecting")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except TimeoutError:
                    pass
                delay = min(delay * 2, self.MAX_RECONNECT_DELAY)
        self.state.set_status("stopped")
        logger.info("[%s] Daemon stopped", self.node_id)

    def snapshot_current_jobs_for_gui(self) -> list[dict[str, Any]]:
        """Best-effort view of the running job for desktop GUI (cross-thread safe enough for UI).

        Returns a list with zero or one entry; workers execute at most one task at a time.
        """
        job = self._current_job
        if not job:
            return []
        job_id = str(job.get("job_id") or "")
        entry_id = job_id or "current"
        kind = job.get("kind")
        if kind == "script":
            tid = str(job.get("task_id") or "").strip()
            name = f"Script ({tid})" if tid else "Script job"
        else:
            name = str(job.get("model_name") or "Training job")
        return [
            {
                "id": entry_id,
                "name": name,
                "priority": "normal",
                "kind": str(kind or "train"),
            }
        ]

    def stop(self) -> None:
        """
        Stop the worker daemon gracefully.

        Cancels running tasks and signals the main loop to exit.

        Example:
            >>> daemon.stop()
        """
        self._stop.set()
        try:
            self.state.flush()
        except OSError as exc:
            logger.warning("[%s] Final state flush failed: %s", self.node_id, exc)
        if self._benchmark_task and not self._benchmark_task.done():
            self._benchmark_task.cancel()
        if self._grpo_round_task and not self._grpo_round_task.done():
            self._grpo_round_task.cancel()
        if self.executor is not None:
            self.executor.cancel()

    async def _connect_and_run(self) -> None:
        jwt_token = _get_jwt_token()
        if not jwt_token and not self._ephemeral:
            auth = self.state.load_auth_tokens()
            jwt_token = auth.jwt_token
        if not jwt_token:
            logger.info(
                "[%s] No JWT token found. Attempting automated registration...", self.node_id
            )
            admin_url = os.getenv("ADMIN_URL") or self._api_base_url()
            reg_mgr = RegistrationManager(admin_url, self.node_id)
            jwt_token = await reg_mgr.register()
            if not jwt_token:
                logger.error(
                    "[%s] Automated registration failed. Please set DISTRIBAI_JWT_TOKEN.",
                    self.node_id,
                )
                await asyncio.sleep(30)
                return
            self.state.save_auth_tokens(jwt_token=jwt_token)
            logger.info("[%s] Automated registration successful ✓", self.node_id)
        logger.info("[%s] Connecting…", self.node_id)
        from services_python.env_bool import env_truthy
        from services_python.grpc_tls import grpc_target_is_public, grpc_tls_enabled

        use_tls = grpc_tls_enabled()
        if grpc_target_is_public(self.orchestrator_url) and not use_tls:
            if env_truthy("ALLOW_INSECURE_PUBLIC_BIND") is not True:
                logger.error(
                    "[%s] Refusing cleartext gRPC to public orchestrator %s; "
                    "set GRPC_USE_TLS=true or ALLOW_INSECURE_PUBLIC_BIND=1 on private nets",
                    self.node_id,
                    self.orchestrator_url,
                )
                await asyncio.sleep(30)
                return
        if use_tls:
            from services_python.grpc_tls import (
                client_ca_path,
                server_cert_path,
                worker_client_cert_path,
                worker_client_key_path,
            )

            root_bytes = None
            ca_path = client_ca_path()
            if ca_path and ca_path.is_file():
                root_bytes = ca_path.read_bytes()
            elif not grpc_target_is_public(self.orchestrator_url):
                dev_cert_path = server_cert_path()
                if dev_cert_path.is_file():
                    root_bytes = dev_cert_path.read_bytes()
            cert_path = worker_client_cert_path()
            key_path = worker_client_key_path()
            if cert_path and key_path and cert_path.is_file() and key_path.is_file():
                credentials = grpc.ssl_channel_credentials(
                    root_certificates=root_bytes,
                    private_key=key_path.read_bytes(),
                    certificate_chain=cert_path.read_bytes(),
                )
            elif root_bytes:
                credentials = grpc.ssl_channel_credentials(root_certificates=root_bytes)
            else:
                credentials = grpc.ssl_channel_credentials()
            channel_ctx = grpc.aio.secure_channel(self.orchestrator_url, credentials)
        else:
            channel_ctx = grpc.aio.insecure_channel(self.orchestrator_url)
        async with channel_ctx as channel:
            stub = distribai_pb2_grpc.NodeServiceStub(channel)
            self.connected = True
            self.state.set_status("connected")
            logger.info("[%s] Connected", self.node_id)
            self._send_queue = asyncio.Queue()

            async def request_generator():
                while True:
                    msg = await self._send_queue.get()
                    if msg is None:
                        break
                    yield msg

            stream = stub.StreamSession(request_generator())
            await self._register()
            hb = asyncio.create_task(self._heartbeat_loop())
            try:
                async for response in stream:
                    await self._handle_message(response)
                    if self._stop.is_set() and not self._current_job:
                        break
            finally:
                hb.cancel()
                await self._send_queue.put(None)
                self.connected = False
                self.state.set_status("disconnected")

    async def _send(self, msg: distribai_pb2.ClientMessage) -> None:
        if self.connected and self._send_queue:
            await self._send_queue.put(msg)
            logger.debug("[%s] → %s", self.node_id, msg.WhichOneof("payload"))

    async def _log_to_orch(self, level: str, message: str) -> None:
        sanitized_msg = _sanitize_log_message(message)
        await self._send(
            distribai_pb2.ClientMessage(
                log=distribai_pb2.LogMessage(
                    node_id=self.node_id,
                    level=level,
                    message=sanitized_msg,
                    ts=int(time.time()),
                )
            )
        )

    def _api_base_url(self) -> str:
        explicit = os.getenv("DISTRIBAI_API_URL")
        if explicit:
            return explicit.rstrip("/")
        host = self.orchestrator_url.split(":", 1)[0]
        scheme = "https" if os.getenv("ADMIN_USE_TLS", "false").lower() == "true" else "http"
        admin_port = os.getenv("ADMIN_PORT", "8766")
        return f"{scheme}://{host}:{admin_port}"

    def _is_jwt_expired(self, token: str) -> bool:
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return True
            payload_b64 = parts[1]
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            exp = payload.get("exp")
            if exp and isinstance(exp, (int, float)):
                return time.time() > exp
            return False
        except (json.JSONDecodeError, ValueError, TypeError, base64.binascii.Error):
            return True

    async def _fetch_or_reuse_jwt(self, hw: dict) -> str:
        env_token = _get_jwt_token()
        if env_token:
            return env_token
        cached = self.state.load_auth_tokens().jwt_token
        if cached and not self._is_jwt_expired(cached):
            return cached
        if cached:
            logger.info("[%s] Cached auth token expired, requesting new token", self.node_id)

        if os.getenv("DISTRIBAI_ALLOW_INSECURE_REGISTER") == "1":
            register_payload = {
                "node_id": self.node_id,
                "public_key": "",
                "invite_code": os.getenv("DISTRIBAI_INVITE_CODE", ""),
                "os": hw.get("os"),
                "gpu_model": hw.get("gpu_model"),
                "driver_version": hw.get("driver_version", ""),
            }
            import aiohttp

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.post(
                    f"{self._api_base_url()}/v1/nodes/register", json=register_payload
                ) as response:
                    body_text = await response.text()
                    if response.status >= 400:
                        raise RuntimeError(
                            f"Node registration failed: {response.status} {body_text}"
                        )
                    try:
                        payload = json.loads(body_text)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Node registration returned invalid JSON: {body_text[:500]}"
                        ) from exc
            self.node_id = payload.get("node_id", self.node_id)
            jwt_token = payload["jwt"]
            self.state.save_auth_tokens(jwt_token=jwt_token)
            return jwt_token

        reg_mgr = RegistrationManager(self._api_base_url(), self.node_id)
        jwt_token = await reg_mgr.register()
        if not jwt_token:
            raise RuntimeError("Node registration failed (PoC challenge flow)")
        self.state.save_auth_tokens(jwt_token=jwt_token)
        return jwt_token

    async def _submit_benchmark_report(
        self, report: dict[str, Any], hardware: dict[str, Any] | None = None
    ) -> None:
        """Submit a completed benchmark through the authenticated node API."""
        hardware = hardware or self._hardware_info()
        token = await self._fetch_or_reuse_jwt(hardware)
        import aiohttp

        payload = {
            **report,
            "driver_version": report.get(
                "driver_version", hardware.get("driver_version", "")
            ),
        }
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(
                f"{self._api_base_url()}/v1/nodes/benchmark",
                json=payload,
                headers=headers,
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(
                        f"benchmark submission failed: {response.status} {body[:500]}"
                    )

    async def _run_startup_benchmark(self, hardware: dict[str, Any]) -> None:
        """Run and submit the first benchmark when no persisted result exists."""
        try:
            report = await self.bench.run_full_suite()
            if report:
                await self._submit_benchmark_report(report, hardware)
                logger.info("[%s] Startup benchmark submitted", self.node_id)
            else:
                logger.warning("[%s] Startup benchmark produced no report", self.node_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[%s] Startup benchmark failed: %s", self.node_id, exc)
        finally:
            if self._benchmark_task is asyncio.current_task():
                self._benchmark_task = None
            if not self._stop.is_set() and not self._current_job:
                self.state.set_status("idle" if self.connected else "disconnected")
            self._maybe_start_deferred_benchmark()

    async def _register(self) -> None:
        hw = self._hardware_info()
        bench_results = self.state.load_benchmark_results()
        if not bench_results:
            logger.info("[%s] No benchmark results found. Running suite now...", self.node_id)
            self.state.set_status("benchmarking")
            if os.getenv("DISTRIBAI_BLOCK_ON_BENCHMARK", "false").lower() == "true":
                bench_results = await self.bench.run_full_suite()
                if bench_results:
                    await self._submit_benchmark_report(bench_results, hw)
            elif not self._benchmark_task or self._benchmark_task.done():
                self._benchmark_task = asyncio.create_task(self._run_startup_benchmark(hw))
            bench_results = bench_results or {}
        jwt_token = await self._fetch_or_reuse_jwt(hw)
        msg = distribai_pb2.ClientMessage(
            register=distribai_pb2.RegisterSession(
                node_id=self.node_id,
                jwt_token=jwt_token,
                hardware_json=json.dumps(hw),
                benchmark_json=json.dumps(bench_results) if bench_results else "",
                ts=int(time.time()),
            )
        )
        await self._send(msg)
        logger.info("[%s] REGISTER sent (including benchmark scores)", self.node_id)

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
                if not self.connected:
                    continue
                self._seq += 1
                hb = distribai_pb2.Heartbeat(
                    node_id=self.node_id,
                    seq=self._seq,
                    vram_free_mb=self._vram_free_mb(),
                    gpu_util=self._gpu_util_pct(),
                    task_id=self._current_job.get("task_id") if self._current_job else None,
                    ts=int(time.time()),
                )
                await self._send(distribai_pb2.ClientMessage(heartbeat=hb))
                self.state.update_heartbeat(self._seq)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("[%s] heartbeat error: %s", self.node_id, e)

    async def _handle_message(self, response: distribai_pb2.ServerMessage) -> None:
        msg_type = response.WhichOneof("payload")
        logger.info("[%s] ← %s", self.node_id, msg_type)
        if msg_type == "register_ack":
            token = response.register_ack.session_token
            if not token:
                logger.error(
                    "[%s] Registration rejected: server returned no session token",
                    self.node_id,
                )
                self._stop.set()
                return
            self._session_token = token
            self.state.save_auth_tokens(session_token=self._session_token)
            logger.info(
                "[%s] Registered ✓ token=%s… Server v%s",
                self.node_id,
                self._session_token[:8],
                response.register_ack.server_version,
            )
            self.state.set_status("idle")
            self._maybe_start_deferred_benchmark()
        elif msg_type == "heartbeat_ack":
            pass
        elif msg_type == "assign":
            await self._accept_job(response.assign)
        elif msg_type == "control":
            action = response.control.action
            logger.info("[%s] Control command received: %s", self.node_id, action)
            await self._handle_control(response.control)
        elif msg_type == "grpo_round_start":
            logger.info("[%s] GRPO round start: job=%s round=%d", self.node_id, response.grpo_round_start.job_id, response.grpo_round_start.round_id)
            await self._handle_grpo_round_start(response.grpo_round_start)
        elif msg_type == "grpo_round_complete":
            logger.info("[%s] GRPO round complete: job=%s round=%d", self.node_id, response.grpo_round_complete.job_id, response.grpo_round_complete.round_id)
            await self._handle_grpo_round_complete(response.grpo_round_complete)
        else:
            logger.warning("[%s] unknown msg type: %s", self.node_id, msg_type)

    def _sanitize_id(self, id_str: str) -> str:
        if not id_str:
            return ""
        sanitized = id_str.replace("/", "_").replace("\\", "_").replace("..", "_")
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in "-_")
        return sanitized[:64]

    async def _accept_job(self, assign: distribai_pb2.TaskAssign) -> None:
        if self._current_job:
            logger.warning("[%s] Already busy, rejecting job %s", self.node_id, assign.job_id)
            await self._send(
                distribai_pb2.ClientMessage(
                    result=distribai_pb2.TaskResult(
                        node_id=self.node_id,
                        job_id=assign.job_id,
                        task_id=assign.task_id,
                        status="rejected",
                        reason="node busy",
                        wall_ms=0,
                        ts=int(time.time()),
                    )
                )
            )
            return
        job_id = self._sanitize_id(assign.job_id)
        task_id = self._sanitize_id(assign.task_id)
        if not job_id or not task_id:
            logger.error("[%s] Rejected job with invalid IDs", self.node_id)
            await self._send(
                distribai_pb2.ClientMessage(
                    result=distribai_pb2.TaskResult(
                        node_id=self.node_id,
                        job_id=assign.job_id,
                        task_id=assign.task_id,
                        status="rejected",
                        reason="invalid job/task ID",
                        wall_ms=0,
                        ts=int(time.time()),
                    )
                )
            )
            return
        if assign.script_package:
            job_id = self._sanitize_id(assign.job_id)
            task_id = self._sanitize_id(assign.task_id)
            self._current_job = {"job_id": job_id, "task_id": task_id, "kind": "script"}
            self.state.set_status("working", job=self._current_job)
            logger.info(
                "[%s] Accepted script job=%s task=%s paradigm=%s",
                self.node_id,
                assign.job_id,
                assign.task_id,
                assign.execution_paradigm or "",
            )
            await self._log_to_orch(
                "info",
                f"Starting script job {assign.job_id} — task={assign.task_id}",
            )
            self._execution_task = asyncio.create_task(
                self._run_script_assign(assign, job_id, task_id)
            )
            return
        job_data = {
            "job_id": job_id,
            "task_id": task_id,
            "model_name": assign.model_name[:128] if assign.model_name else "unknown",
            "weight_blob_url": assign.weight_blob_url,
            "batch_blob_url": assign.batch_blob_url,
            "hparams": json.loads(assign.hparams_json) if assign.hparams_json else {},
            "steps": max(1, min(assign.steps, 10000)),
            "batch_size": max(
                1,
                min(
                    int(json.loads(assign.hparams_json).get("batch_size", 32))
                    if assign.hparams_json
                    else 32,
                    2048,
                ),
            ),
            "deadline_ts": int(assign.deadline_ts)
            if assign.deadline_ts
            else int(time.time()) + 600,
            "weight_version": assign.weight_version or "",
        }
        self._current_job = job_data
        self.state.set_status("working", job=job_data)
        logger.info(
            "[%s] Accepted job=%s model=%s steps=%s",
            self.node_id,
            assign.job_id,
            assign.model_name,
            assign.steps,
        )
        await self._log_to_orch(
            "info",
            f"Starting job {assign.job_id} — model={assign.model_name}, steps={assign.steps}",
        )
        self._execution_task = asyncio.create_task(self._get_executor().execute(job_data))

    async def _run_script_assign(
        self,
        assign: distribai_pb2.TaskAssign,
        job_id: str,
        task_id: str,
    ) -> None:
        from .script_runner import ScriptRunner

        t0 = time.time()
        merged_env: dict[str, str] = {}
        if assign.distributed_env_json:
            try:
                parsed = json.loads(assign.distributed_env_json)
                if isinstance(parsed, dict):
                    merged_env = {str(k): str(v) for k, v in parsed.items()}
            except json.JSONDecodeError:
                logger.warning("[%s] invalid distributed_env_json", self.node_id)

        hyperparams: dict[str, Any] = {}
        if assign.hparams_json:
            try:
                loaded = json.loads(assign.hparams_json)
                if isinstance(loaded, dict):
                    hyperparams = loaded
            except json.JSONDecodeError:
                logger.warning("[%s] invalid hparams_json for script job", self.node_id)

        self._script_runner = ScriptRunner()
        try:
            result = await self._script_runner.execute_task(
                task_id,
                bytes(assign.script_package),
                merged_env,
                hyperparams,
            )
        except asyncio.CancelledError:
            if self._script_runner:
                self._script_runner.cancel_task(task_id)
            raise
        finally:
            self._script_runner = None

        wall_ms = int((time.time() - t0) * 1000)
        ok = result.get("status") == "completed"
        out = dict(result)
        if not ok and out.get("error") is None and out.get("stderr"):
            out["error"] = str(out.get("stderr", ""))[:4000]
        status = "success" if ok else "failed"
        await self._report_result(job_id, task_id, status, wall_ms, out)

    def _maybe_start_deferred_benchmark(self, *, allow_disconnected: bool = False) -> None:
        """Start a deferred benchmark once the worker is connected and idle."""
        if (
            self._benchmark_requested
            and (self.connected or allow_disconnected)
            and not self._current_job
            and (not self._benchmark_task or self._benchmark_task.done())
            and not self._stop.is_set()
        ):
            self._benchmark_requested = False
            self._benchmark_task = asyncio.create_task(self._run_requested_benchmark())

    async def _handle_control(self, control: distribai_pb2.ControlMessage) -> None:
        action = control.action
        if action == "pause":
            if self._current_job and self._current_job.get("kind") == "script":
                logger.info("[%s] pause not applied to script job", self.node_id)
            else:
                await self._get_executor().pause()
                self.state.set_status("paused")
        elif action == "resume":
            if self._current_job and self._current_job.get("kind") == "script":
                logger.info("[%s] resume not applied to script job", self.node_id)
            else:
                await self._get_executor().resume()
                self.state.set_status("working" if self._current_job else "idle")
        elif action == "drain":
            logger.info("[%s] Draining — will stop after current job", self.node_id)
            self._stop.set()
        elif action == "cancel_job":
            target = control.target_id
            if self._current_job and self._current_job.get("job_id") == target:
                logger.info("[%s] Cancelling job %s", self.node_id, target)
                if self._current_job.get("kind") == "script" and self._script_runner:
                    self._script_runner.cancel_task(self._current_job.get("task_id", ""))
                else:
                    self._get_executor().cancel()
                if self._execution_task and not self._execution_task.done():
                    self._execution_task.cancel()
        elif action == "benchmark":
            if self._benchmark_task and not self._benchmark_task.done():
                self._benchmark_requested = True
                logger.info(
                    "[%s] Benchmark already running; request retained for the next idle window",
                    self.node_id,
                )
            elif self._current_job:
                self._benchmark_requested = True
                logger.info(
                    "[%s] Benchmark deferred until current %s job completes",
                    self.node_id,
                    self._current_job.get("kind", "training"),
                )
            else:
                self._benchmark_requested = True
                self._maybe_start_deferred_benchmark(allow_disconnected=True)
        elif action == "bft_aggregate_ready":
            target = control.target_id
            logger.info("[%s] BFT aggregate ready for job %s", self.node_id, target)
            if self._current_job and self._current_job.get("job_id") == target:
                if self._current_job.get("kind") == "script":
                    return
                try:
                    await self._get_executor().on_aggregate_ready(target)
                except Exception as exc:
                    logger.warning("on_aggregate_ready failed: %s", exc)

    async def _run_requested_benchmark(self) -> None:
        """Run a benchmark requested by the orchestrator and submit its report."""
        if self._current_job:
            self._benchmark_requested = True
            logger.info(
                "[%s] Benchmark remains deferred while %s job is active",
                self.node_id,
                self._current_job.get("kind", "training"),
            )
            return

        self.state.set_status("benchmarking")
        try:
            report = await self.bench.run_full_suite()
            if not report:
                logger.warning("[%s] Requested benchmark produced no report", self.node_id)
                return
            await self._submit_benchmark_report(report)
            logger.info("[%s] Requested benchmark submitted", self.node_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[%s] Requested benchmark failed: %s", self.node_id, exc)
        finally:
            if self._benchmark_task is asyncio.current_task():
                self._benchmark_task = None
            if not self._stop.is_set() and not self._current_job:
                self.state.set_status("idle" if self.connected else "disconnected")
            self._maybe_start_deferred_benchmark()

    # ── GRPO handlers ──────────────────────────────────────────────────

    async def _ensure_grpo_runner(self) -> Any:
        """Lazy-initialize the GrpoRunner."""
        if self._grpo_runner is None:
            from .grpo_runner import GrpoRunner, GrpoRunnerConfig

            # Build a minimal model and tokenizer for GRPO
            model = self._get_grpo_model()
            tokenizer = self._get_grpo_tokenizer()

            cfg = GrpoRunnerConfig(
                job_id="",
                worker_id=self.node_id,
            )

            self._grpo_runner = GrpoRunner(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                device="cuda" if torch.cuda.is_available() else "cpu",
                send_reward_report=self._send_grpo_reward_report,
            )
        return self._grpo_runner

    def _get_grpo_model(self) -> torch.nn.Module:
        """Create the native DistribAI policy model used for GRPO training."""
        from ..compute.distribai_models import get_model

        model = get_model("distribai-tiny", vocab_size=256)
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def _get_grpo_tokenizer(self) -> Any:
        """Return a tokenizer for GRPO candidate generation."""
        # Use a simple byte-level tokenizer as default
        class _SimpleTokenizer:
            def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
                ids = list(text.encode("utf-8"))
                return ids[:256]  # truncate
            def decode(self, ids: list[int], skip_special: bool = False) -> str:
                return bytes(ids).decode("utf-8", errors="replace")
            @property
            def eos_id(self) -> int:
                return 0

        return _SimpleTokenizer()

    async def _handle_grpo_round_start(self, msg: Any) -> None:
        """Handle GrpoRoundStart: download weights + prompts, generate candidates, report rewards."""
        runner = await self._ensure_grpo_runner()

        # Update runner config from the GrpoConfig protobuf
        cfg = msg.config
        runner.cfg.job_id = msg.job_id
        runner.cfg.group_size = max(1, cfg.group_size)
        runner.cfg.kl_coef = cfg.kl_coef
        runner.cfg.clip_eps = cfg.clip_eps
        runner.cfg.reward_scale = cfg.reward_scale
        runner.cfg.prompts_per_step = max(1, cfg.prompts_per_step)
        runner.cfg.max_gen_tokens = max(1, cfg.max_gen_tokens)
        runner.cfg.gen_temperature = max(0.1, cfg.gen_temperature)
        runner.cfg.gen_top_k = max(1, cfg.gen_top_k)
        runner.cfg.ref_model_url = cfg.ref_model_url

        grpo_config_dict = {
            "group_size": runner.cfg.group_size,
            "kl_coef": runner.cfg.kl_coef,
            "clip_eps": runner.cfg.clip_eps,
            "reward_scale": runner.cfg.reward_scale,
            "prompts_per_step": runner.cfg.prompts_per_step,
            "max_gen_tokens": runner.cfg.max_gen_tokens,
            "gen_temperature": runner.cfg.gen_temperature,
            "gen_top_k": runner.cfg.gen_top_k,
        }

        # Set current job so heartbeats show we're working
        self._current_job = {
            "job_id": msg.job_id,
            "task_id": f"grpo_round_{msg.round_id}",
            "kind": "grpo",
            "model_name": "grpo_policy",
        }
        self.state.set_status("working", job=self._current_job)

        # Start round in runner (non-blocking)
        def _round_start_sync():
            runner.on_round_start(
                round_id=msg.round_id,
                weights_blob_url=msg.weights_blob_url,
                prompts_json_url=msg.prompts_json_url,
                grpo_config=grpo_config_dict,
            )

        await asyncio.to_thread(_round_start_sync)

        # Generate candidates and report rewards (this may be CPU/GPU intensive)
        self._grpo_round_task = asyncio.create_task(
            self._run_grpo_candidate_generation(msg.job_id, msg.round_id)
        )

    async def _run_grpo_candidate_generation(self, job_id: str, round_id: int) -> None:
        """Run candidate generation in the background."""
        try:
            runner = self._grpo_runner
            if runner is None:
                return

            result = await asyncio.to_thread(runner.generate_candidates_and_report)
            logger.info(
                "[%s] GRPO round %d: generated %d candidates (mean reward=%.4f)",
                self.node_id,
                round_id,
                result.get("num_candidates", 0),
                result.get("mean_reward", 0.0),
            )
        except Exception as exc:
            logger.error(
                "[%s] GRPO round %d candidate generation failed: %s",
                self.node_id,
                round_id,
                exc,
                exc_info=True,
            )

    async def _handle_grpo_round_complete(self, msg: Any) -> None:
        """Handle GrpoRoundComplete: apply GRPO update and load new weights."""
        runner = self._grpo_runner
        if runner is None:
            logger.warning("[%s] GRPO round complete but no runner", self.node_id)
            return

        # Candidate generation runs in a worker thread and mutates runner state.
        # Await it before applying new weights so generation and update never
        # access the runner concurrently.
        candidate_task = self._grpo_round_task
        if candidate_task and not candidate_task.done():
            try:
                await candidate_task
            except asyncio.CancelledError:
                logger.info("[%s] GRPO candidate generation was cancelled", self.node_id)

        # Parse advantages
        advantages: list[float] = []
        if msg.advantages_json:
            try:
                parsed = json.loads(msg.advantages_json)
                if isinstance(parsed, list):
                    advantages = [float(v) for v in parsed]
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning("[%s] Invalid advantages_json: %s", self.node_id, exc)

        def _round_complete_sync():
            return runner.on_round_complete(
                round_id=msg.round_id,
                new_weights_blob_url=msg.new_weights_blob_url,
                advantages=advantages,
            )

        try:
            metrics = await asyncio.to_thread(_round_complete_sync)

            # Report progress to orchestrator
            await self._report_progress(
                job_id=runner.cfg.job_id,
                task_id=f"grpo_round_{msg.round_id}",
                step=msg.round_id,
                loss=metrics.get("total_loss", 0.0),
            )

            logger.info(
                "[%s] GRPO round %d complete: policy_loss=%.4f kl_loss=%.4f grad_norm=%.4f",
                self.node_id,
                msg.round_id,
                metrics.get("policy_loss", 0.0),
                metrics.get("kl_loss", 0.0),
                metrics.get("grad_norm", 0.0),
            )
        except Exception as exc:
            logger.error(
                "[%s] GRPO round %d complete failed: %s",
                self.node_id,
                msg.round_id,
                exc,
                exc_info=True,
            )
        finally:
            self._grpo_round_task = None
            if self._current_job and self._current_job.get("kind") == "grpo":
                self._current_job = None
                if not self._stop.is_set():
                    self.state.set_status("idle" if self.connected else "disconnected")
                self._maybe_start_deferred_benchmark()

    def _send_grpo_reward_report(
        self,
        job_id: str,
        round_id: int,
        rewards: list[float],
        texts_json: str | None,
    ) -> None:
        """Callback used by GrpoRunner to send reward reports to the orchestrator."""
        msg = distribai_pb2.ClientMessage(
            grpo_reward_report=distribai_pb2.GrpoRewardReport(
                job_id=job_id,
                round_id=round_id,
                worker_id=self.node_id,
                candidate_rewards=rewards,
                candidate_texts_json=texts_json or "",
                ts=int(time.time()),
            )
        )

        # Fire-and-forget via the send queue
        if self.connected and self._send_queue:
            try:
                asyncio.ensure_future(self._send(msg))
            except Exception as exc:
                logger.error("[%s] Failed to send GRPO reward report: %s", self.node_id, exc)

    async def _report_progress(self, job_id: str, task_id: str, step: int, loss: float) -> None:
        now = time.time()
        if (
            now - self._last_progress_report < PROGRESS_THROTTLE_SECONDS
            and step % PROGRESS_MILESTONE_STEPS != 0
        ):
            return
        self._last_progress_report = now
        await self._send(
            distribai_pb2.ClientMessage(
                progress=distribai_pb2.TaskProgress(
                    node_id=self.node_id,
                    job_id=job_id,
                    task_id=task_id,
                    step=step,
                    loss=float(loss),
                    ts=int(time.time()),
                )
            )
        )

    async def _report_result(
        self,
        job_id: str,
        task_id: str,
        status: str,
        wall_ms: int,
        output: dict,
    ) -> None:
        gradient_blob_url = output.get("gradient_blob_url", "") if isinstance(output, dict) else ""
        reason = output.get("error", "") if isinstance(output, dict) else ""
        self._execution_task = None
        self._current_job = None
        self.state.set_status("idle")
        self.state.record_job_done(status)
        await self._send(
            distribai_pb2.ClientMessage(
                result=distribai_pb2.TaskResult(
                    node_id=self.node_id,
                    job_id=job_id,
                    task_id=task_id,
                    status=status,
                    gradient_blob_url=gradient_blob_url,
                    wall_ms=wall_ms,
                    reason=reason,
                    output_json=json.dumps(output),
                    ts=int(time.time()),
                )
            )
        )
        await self._log_to_orch(
            "info" if status == "success" else "warning",
            f"Job {job_id} finished status={status} wall={wall_ms}ms",
        )
        self._maybe_start_deferred_benchmark()

    def _hardware_info(self) -> dict:
        mem = psutil.virtual_memory()
        mac = hex(uuid.getnode())
        return {
            "os": platform.system(),
            "hostname": socket.gethostname(),
            "hwid_mac": mac,
            "cpu_cores": os.cpu_count() or 1,
            "ram_gb": round(mem.total / (1024**3), 1),
            "gpu_model": self._gpu_name(),
            "vram_mb": self._vram_total_mb(),
            "python_version": platform.python_version(),
            "worker_index": self.worker_index,
        }

    @staticmethod
    def _run_nvidia(query: str) -> str:
        allowed_queries = {"name", "memory.total", "memory.free", "utilization.gpu"}
        if query not in allowed_queries:
            logger.warning(f"Invalid nvidia-smi query: {query}")
            return ""
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return ""
        run_kw: dict[str, Any] = {}
        if platform.system() == "Windows":
            # Avoid console windows and stuck stdin pipes when drivers misbehave.
            run_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            r = subprocess.run(
                [nvidia_smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=3,
                check=False,
                **run_kw,
            )
            if r.returncode == 0:
                return r.stdout.strip()
        except (
            subprocess.SubprocessError,
            subprocess.TimeoutExpired,
            FileNotFoundError,
            OSError,
        ):
            logger.debug("nvidia-smi query failed for %s", query)
        return ""

    def _gpu_name(self) -> str:
        v = self._run_nvidia("name")
        return v.split("\n")[0] if v else "CPU-only"

    def _vram_total_mb(self) -> int:
        v = self._run_nvidia("memory.total")
        try:
            return int(v.split("\n")[0])
        except (ValueError, IndexError):
            return 0

    def _vram_free_mb(self) -> int:
        v = self._run_nvidia("memory.free")
        try:
            return int(v.split("\n")[0])
        except (ValueError, IndexError):
            return 0

    def _gpu_util_pct(self) -> float:
        v = self._run_nvidia("utilization.gpu")
        try:
            return float(v.split("\n")[0])
        except (ValueError, IndexError):
            return 0.0

    @staticmethod
    def _make_node_id(worker_index: int) -> str:
        suffix = secrets.token_hex(8)
        host = socket.gethostname().split(".")[0]
        if worker_index > 0:
            return f"{host}-w{worker_index:02d}-{suffix}"
        return f"{host}-{suffix}"
