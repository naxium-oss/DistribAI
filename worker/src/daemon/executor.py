"""
Per-task training executor used by the worker daemon.
"""

import asyncio
import gc
import hashlib
import json
import logging
import os
import platform
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as functional

from services_python.blob_loader import load_text_blob
from services_python.blob_url_policy import is_allowed_gradient_url

from .oom_guard import OOMGuard, OOMStrategy
from .optimizers import build_optimizer


def _resolve_default_optimizer() -> str:
    raw = os.getenv("DISTRIBAI_DEFAULT_OPTIMIZER", "auon").strip().lower()
    return raw or "auon"


def _resolve_job_optimizer(hyperparams: dict[str, Any]) -> str:
    override = hyperparams.get("optimizer")
    if override:
        return str(override).strip().lower()
    return _resolve_default_optimizer()


def get_resource_limits() -> dict[str, int]:
    """Read cpu/gpu/ram percent caps from ~/.distribai/desktop.json.

    Returns:
        Mapping of cpuPercent, gpuPercent, ramPercent (clamped 10-100).
    """
    try:
        config_path = Path.home() / ".distribai" / "desktop.json"
        if config_path.exists():
            data = json.loads(config_path.read_text())
            return {
                "cpuPercent": max(10, min(100, data.get("cpuPercent", 50))),
                "gpuPercent": max(10, min(100, data.get("gpuPercent", 50))),
                "ramPercent": max(10, min(100, data.get("ramPercent", 50))),
            }
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logging.getLogger(__name__).debug("Could not load resource limits: %s", e)
    return {"cpuPercent": 50, "gpuPercent": 50, "ramPercent": 50}


def apply_cpu_limit(percent: int) -> None:
    """Constrain CPU via affinity mask (and nice on POSIX).

    Args:
        percent: Target CPU share from 10 to 100.
    """
    if percent >= 100:
        return

    try:
        process = psutil.Process()
        total_cores = psutil.cpu_count(logical=True)

        # Core count = ceil of percent of logical CPUs
        cores_to_use = max(1, int(total_cores * percent / 100))
        affinity_mask = list(range(cores_to_use))

        # Restrict the process to the first N cores
        process.cpu_affinity(affinity_mask)

        # POSIX: raise nice so peers get more scheduler share
        if platform.system() != "Windows":
            # Higher nice value yields lower relative priority
            nice_value = min(19, int((100 - percent) / 5))
            os.nice(nice_value)

        logging.getLogger(__name__).info(
            "CPU limited to %s%% (using %s/%s cores)", percent, cores_to_use, total_cores
        )
    except Exception as e:
        logging.getLogger(__name__).warning("Could not apply CPU limit: %s", e)


def apply_gpu_limit(percent: int) -> None:
    """Cap process VRAM via torch.cuda.set_per_process_memory_fraction.

    Args:
        percent: Allowed VRAM share from 10 to 100.
    """
    if percent >= 100 or not torch.cuda.is_available():
        return

    try:
        # Total device memory for logging
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Soft byte cap derived from percent
        allowed_memory = int(total_memory * percent / 100)

        # Enforce fraction with the CUDA allocator
        torch.cuda.set_per_process_memory_fraction(percent / 100)

        logging.getLogger(__name__).info(
            f"GPU memory limited to {percent}% ({allowed_memory // (1024**2)} MB)"
        )
    except Exception as e:
        logging.getLogger(__name__).warning(f"Could not apply GPU limit: {e}")


class ResourceMonitor:
    """Watch RSS against the configured RAM cap and relieve pressure."""

    def __init__(self, ram_percent: int) -> None:
        self.ram_percent = ram_percent
        self.process = psutil.Process()
        self.total_ram = psutil.virtual_memory().total
        self.max_ram = int(self.total_ram * ram_percent / 100)
        self._stop_event = asyncio.Event()
        self._monitor_task: asyncio.Task | None = None
        self._memory_pressure_event = asyncio.Event()
        self._memory_pressure_start = None
        self._pressure_duration_limit = 30  # seconds
        self._last_cleanup = 0
        self._cleanup_interval = 10  # seconds

    async def start(self) -> None:
        """Spawn the async RSS monitor task."""
        self._stop_event.clear()
        self._monitor_task = asyncio.create_task(self._monitor())

    async def stop(self) -> None:
        """Signal the monitor loop to exit and await it."""
        self._stop_event.set()
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    def is_memory_pressure_active(self) -> bool:
        """True while RSS sits above the configured ceiling."""
        return self._memory_pressure_event.is_set()

    async def wait_for_memory_pressure_relief(self, timeout: float = 60.0) -> bool:
        """Block until pressure clears or timeout elapses."""
        try:
            await asyncio.wait_for(self._memory_pressure_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def _monitor(self) -> None:
        """Loop: sample RSS, escalate pressure, run relief helpers."""
        while not self._stop_event.is_set():
            try:
                current_ram = self.process.memory_info().rss
                current_time = time.time()

                if current_ram > self.max_ram:
                    # RSS above max_ram
                    if not self._memory_pressure_event.is_set():
                        self._memory_pressure_event.set()
                        self._memory_pressure_start = current_time
                        logging.getLogger(__name__).warning(
                            f"RAM usage ({current_ram // (1024**2)}MB) exceeded limit "
                            f"({self.max_ram // (1024**2)}MB), applying pressure relief"
                        )

                    # Log harder if pressure outlives the duration limit
                    if (
                        self._memory_pressure_start
                        and current_time - self._memory_pressure_start
                        > self._pressure_duration_limit
                    ):
                        logging.getLogger(__name__).error(
                            f"Memory pressure persisted for {self._pressure_duration_limit}s, "
                            "consider reducing resource limits or batch size"
                        )

                    # Trigger GC and device cache drops
                    await self._apply_memory_relief()

                    # Slow the poll cadence during relief
                    await asyncio.sleep(2)
                else:
                    # Back under the cap
                    if self._memory_pressure_event.is_set():
                        self._memory_pressure_event.clear()
                        self._memory_pressure_start = None
                        logging.getLogger(__name__).info(
                            f"Memory pressure relieved, usage: {current_ram // (1024**2)}MB"
                        )

                    # Periodic cheap cleanup when idle
                    if current_time - self._last_cleanup > self._cleanup_interval:
                        await self._periodic_cleanup()
                        self._last_cleanup = current_time

                    await asyncio.sleep(1)

            except (OSError, psutil.Error) as e:
                logging.getLogger(__name__).debug(f"Resource monitoring error: {e}")
                await asyncio.sleep(5)

    async def _apply_memory_relief(self) -> None:
        """GC plus CUDA/MPS cache eviction while under pressure."""
        try:
            # Force a full garbage collection
            gc.collect()

            # Free idle CUDA caching allocator blocks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.getLogger(__name__).debug("Cleared CUDA cache")

            # Free Apple MPS cache when available
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
                logging.getLogger(__name__).debug("Cleared MPS cache")

            # Touch process memory_info to refresh counters
            try:
                self.process.memory_info()  # Refresh memory info
            except (OSError, psutil.Error):
                pass

        except Exception as e:
            logging.getLogger(__name__).warning(f"Memory relief failed: {e}")

    async def _periodic_cleanup(self) -> None:
        """Light GC (and conditional CUDA empty) on the quiet path."""
        try:
            # Inexpensive collect when not pressured
            gc.collect()

            # Empty CUDA cache when nearing 80% of max_ram
            if torch.cuda.is_available():
                # Soft threshold: 80% of configured max
                current_usage = self.process.memory_info().rss
                if current_usage > self.max_ram * 0.8:
                    torch.cuda.empty_cache()
                    logging.getLogger(__name__).debug("Periodic CUDA cache cleanup")

        except Exception as e:
            logging.getLogger(__name__).debug(f"Periodic cleanup failed: {e}")


try:
    from ..compute import ComputeBackend, detect_backend
    from ..compute.distribai_models import CustomModelBuilder, DistribAIModelWrapper, get_model
    from .gradient_compression import DeepGradientCompression
    from .s3_util import S3Manager
except ImportError:
    try:
        from compute import ComputeBackend, detect_backend
        from compute.distribai_models import CustomModelBuilder, DistribAIModelWrapper, get_model
        from daemon.gradient_compression import DeepGradientCompression
        from daemon.s3_util import S3Manager
    except ImportError:
        import sys
        from pathlib import Path

        _src_dir = Path(__file__).resolve().parent.parent
        if str(_src_dir) not in sys.path:
            sys.path.insert(0, str(_src_dir))
        from compute import ComputeBackend, detect_backend
        from compute.distribai_models import CustomModelBuilder, DistribAIModelWrapper, get_model
        from daemon.gradient_compression import DeepGradientCompression
        from daemon.s3_util import S3Manager
logger = logging.getLogger(__name__)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class JobExecutor:
    """
    Job Executor for DistribAI Worker

    Handles execution of training tasks on worker nodes including
    gradient computation, checkpoint management, and result reporting.
    """

    def __init__(
        self,
        node_id: str,
        on_progress: Callable[[str, str, int, float], Awaitable[None]],
        on_result: Callable[[str, str, str, int, dict], Awaitable[None]],
    ) -> None:
        """
        Initialize the job executor.

        Args:
            node_id: Unique identifier for this worker node
            on_progress: Optional callback for progress updates (step, total)
            on_result: Optional callback for result reporting

        Example:
            >>> executor = JobExecutor(
            ...     node_id="worker-001",
            ...     on_progress=lambda s, t: print(f"{s}/{t} steps")
            ... )
        """
        self.node_id = node_id
        self.on_progress = on_progress
        self.on_result = on_result
        self._paused = asyncio.Event()
        self._paused.set()
        self._pause_lock = asyncio.Lock()
        self._cancelled = False
        self._current_task: asyncio.Task | None = None

        # Wire OOM guard from env (strategy + retry budget)
        oom_strategy_str = os.getenv("DISTRIBAI_OOM_STRATEGY", "reduce_batch_size")
        self.oom_guard = OOMGuard(
            strategy=OOMStrategy(oom_strategy_str),
            max_retries=int(os.getenv("DISTRIBAI_OOM_MAX_RETRIES", "3")),
            enable_telemetry=True,
        )

        self.s3 = S3Manager()
        self.gradient_compressor = DeepGradientCompression(
            sparsity=float(os.getenv("DISTRIBAI_GRADIENT_SPARSITY", "0.9"))
        )
        self.backend: ComputeBackend | None = None
        try:
            self.backend = detect_backend()
            if self.backend:
                logger.info(f"[{node_id}] Using compute backend: {self.backend.name}")
            else:
                logger.warning(f"[{node_id}] No compute backend available, falling back to CPU")
                from ..compute.cpu import CPUBackend

                self.backend = CPUBackend()
                self.backend.initialize()
        except Exception as e:
            logger.error(f"[{node_id}] Failed to initialize compute backend: {e}")
            self.backend = None

    def _create_model(
        self,
        model_name: str,
        architecture_config: dict[str, Any] | None = None,
        hyperparams: dict[str, Any] | None = None,
    ) -> nn.Module:
        model_name_lower = model_name.lower()
        from worker.src.compute.distribai_models import DistribAIModelWrapper
        from worker.src.compute.external_arch import (
            load_external_architecture,
            looks_like_external_model_ref,
        )

        hp = hyperparams or {}
        if architecture_config is not None:
            logger.info(
                "Creating uploaded DistribAI architecture family %s",
                architecture_config.get("family", architecture_config.get("architecture")),
            )
            return get_model(
                model_name_lower or "uploaded-architecture",
                vocab_size=256,
                architecture_config=architecture_config,
            )
        external_ref = (
            hp.get("external_model")
            or hp.get("hf_model_id")
            or hp.get("hf_repo")
            or (model_name if looks_like_external_model_ref(model_name) else None)
        )
        if external_ref:
            allow = hp.get("allow_external_arch")
            if allow is None:
                allow = hp.get("trust_remote_code")
            logger.info("Creating external architecture from %s", external_ref)
            config_overrides = hp.get("config_overrides")
            return load_external_architecture(
                str(external_ref),
                trust_remote_code=bool(hp.get("trust_remote_code", True)),
                torch_dtype=hp.get("torch_dtype"),
                allow=allow if isinstance(allow, bool) else None,
                config_overrides=config_overrides if isinstance(config_overrides, dict) else None,
                from_scratch=bool(hp.get("from_scratch", False)),
            )
        if model_name_lower in DistribAIModelWrapper.MODEL_CONFIGS:
            logger.info("Creating native DistribAI model profile %s", model_name_lower)
            return get_model(model_name_lower, vocab_size=256)
        if model_name_lower == "custom":
            logger.info("Creating explicitly requested custom DistribAI model")
            return CustomModelBuilder.create_custom_model(vocab_size=256)
        if model_name_lower in {"tiny", "small", "medium"}:
            return get_model(model_name_lower, vocab_size=256)
        if model_name_lower == "toy" and os.getenv("DISTRIBAI_ALLOW_TEST_MODELS") == "1":
            logger.warning("Creating test-only ToyModel because DISTRIBAI_ALLOW_TEST_MODELS=1")
            return ToyModel()
        raise ValueError(
            f"Unknown model profile {model_name!r}; available profiles: "
            f"{sorted(DistribAIModelWrapper.MODEL_CONFIGS)} plus custom, "
            "or an external Hub/local architecture reference"
        )

    async def pause(self) -> None:
        async with self._pause_lock:
            self._paused.clear()
            logger.info("[%s] executor paused", self.node_id)

    async def resume(self) -> None:
        async with self._pause_lock:
            self._paused.set()
            logger.info("[%s] executor resumed", self.node_id)

    async def on_aggregate_ready(self, job_id: str) -> None:
        """Orchestrator persisted a BFT aggregate for this job."""
        logger.info(
            "[%s] Server BFT aggregate ready for job=%s (worker continues training loop)",
            self.node_id,
            job_id,
        )

    def cancel(self) -> None:
        self._cancelled = True
        self._paused.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    async def execute(self, job: dict[str, Any]) -> None:
        self._current_task = asyncio.current_task()
        self._cancelled = False
        job_id = job["job_id"]
        task_id = job["task_id"]
        steps = max(1, int(job.get("steps", 50)))
        batch_size = max(1, int(job.get("batch_size", 32)))
        model_name = str(job.get("model_name", "distribai-small"))
        deadline_ts = int(job.get("deadline_ts", time.time() + 600))
        weight_url = job.get("weight_blob_url")
        batch_blob_url = job.get("batch_blob_url")
        # The gRPC task-assign path (daemon._accept_job) stores this under
        # "hparams" (matching the wire field hparams_json); direct callers/tests
        # commonly use "hyperparams". Accept either so architecture_config and
        # other job-level knobs always reach model creation.
        hyperparams = job.get("hyperparams") or job.get("hparams") or {}
        architecture_config = hyperparams.get("architecture_config")
        optimizer_name = _resolve_job_optimizer(hyperparams)
        progress_every = max(1, steps // 10)
        start = time.monotonic()

        # Read desktop caps and apply CPU/GPU limits
        resource_limits = get_resource_limits()
        logger.info(
            f"[{self.node_id}] Resource limits: CPU={resource_limits['cpuPercent']}%, "
            f"GPU={resource_limits['gpuPercent']}%, RAM={resource_limits['ramPercent']}%"
        )

        apply_cpu_limit(resource_limits["cpuPercent"])
        apply_gpu_limit(resource_limits["gpuPercent"])

        # Background RSS watchdog for this job
        ram_monitor = ResourceMonitor(resource_limits["ramPercent"])
        await ram_monitor.start()

        try:
            try:
                with self.oom_guard.guard("model_creation"):
                    if architecture_config is None and not (
                        (hyperparams or {}).get("external_model")
                        or (hyperparams or {}).get("hf_model_id")
                        or (hyperparams or {}).get("hf_repo")
                    ):
                        # Keep the one-arg call path for callers that patch
                        # or subclass _create_model(model_name) only.
                        model = self._create_model(model_name)
                    else:
                        model = self._create_model(
                            model_name,
                            architecture_config=architecture_config,
                            hyperparams=hyperparams,
                        )
            except Exception as create_exc:
                logger.error(
                    "[%s] job=%s model creation failed: %s",
                    self.node_id,
                    job_id,
                    create_exc,
                    exc_info=True,
                )
                wall_ms = int((time.monotonic() - start) * 1000)
                await self.on_result(
                    job_id,
                    task_id,
                    "error",
                    wall_ms,
                    {"error": str(create_exc), "model_name": model_name},
                )
                return
            if self.backend:
                try:
                    with self.oom_guard.guard("model_optimization"):
                        model = self.backend.optimize_model(model)
                    logger.info(f"[{self.node_id}] Model optimized for {self.backend.name}")
                except Exception as e:
                    logger.warning(f"[{self.node_id}] Model optimization failed: {e}")
            try:
                await self._load_weights(model, task_id, weight_url)
                batch_source = await self._load_batch_source(task_id, batch_blob_url)
            except Exception as prep_exc:
                logger.error("[%s] job=%s prep failed: %s", self.node_id, job_id, prep_exc, exc_info=True)
                wall_ms = int((time.monotonic() - start) * 1000)
                await self.on_result(job_id, task_id, "error", wall_ms, {"error": str(prep_exc)})
                return
            if self._is_language_model(model):
                model_config = getattr(model, "config", None)
                configured_seq_len = getattr(model_config, "seq_len", None) or getattr(
                    model_config, "max_position_embeddings", None
                )
                train_batch = self._build_language_model_batch(
                    batch_source, batch_size, task_id, seq_len=configured_seq_len
                )
            else:
                train_batch = self._build_toy_batch(batch_source, batch_size, task_id)
            lr = 0.001 if self._is_language_model(model) else 0.01
            optimizer = build_optimizer(optimizer_name, model.parameters(), lr=lr)
            initial_loss = 0.0
            loss_val = 0.0
            try:
                for step in range(1, steps + 1):
                    await self._paused.wait()

                    # Pause the step loop while RSS pressure is active
                    if ram_monitor.is_memory_pressure_active():
                        logger.info(f"[{self.node_id}] Waiting for memory pressure relief...")
                        if not await ram_monitor.wait_for_memory_pressure_relief(timeout=30.0):
                            logger.warning(
                                f"[{self.node_id}] Memory pressure persists, continuing with reduced performance"
                            )

                    if self._cancelled:
                        raise asyncio.CancelledError
                    if int(time.time()) > deadline_ts:
                        await self.on_result(
                            job_id,
                            task_id,
                            "timeout",
                            int((time.monotonic() - start) * 1000),
                            {"error": "deadline exceeded"},
                        )
                        return
                    with self.oom_guard.guard("training_step"):
                        optimizer.zero_grad()
                        loss = self._compute_loss(model, train_batch)
                        loss.backward()
                        optimizer.step()
                    if self.backend:
                        self.backend.synchronize()
                    loss_val = float(loss.item())
                    if step == 1:
                        initial_loss = loss_val
                    if step % progress_every == 0 or step == steps:
                        await self.on_progress(job_id, task_id, step, loss_val)
                    await asyncio.sleep(0)
                gradients, gradient_norm = self._collect_gradients(model)
                gradients_path = self._write_gradients(task_id, gradients)
                gradient_key = f"gradients/{job_id}/{gradients_path.name}"
                gradient_url = await self.s3.upload_file(str(gradients_path), gradient_key)
                wall_ms = int((time.monotonic() - start) * 1000)
                await self.on_result(
                    job_id,
                    task_id,
                    "success",
                    wall_ms,
                    {
                        "final_loss": round(loss_val, 6),
                        "initial_loss": round(initial_loss, 6),
                        "steps_completed": steps,
                        "gradient_blob_url": gradient_url,
                        "model_name": model_name,
                        "gradient_norm": round(gradient_norm, 6),
                        "weight_version": job.get("weight_version"),
                    },
                )
            except asyncio.CancelledError:
                wall_ms = int((time.monotonic() - start) * 1000)
                await self.on_result(
                    job_id, task_id, "cancelled", wall_ms, {"error": "task cancelled"}
                )
                raise
            except Exception as exc:
                logger.error("[%s] job=%s error: %s", self.node_id, job_id, exc, exc_info=True)
                wall_ms = int((time.monotonic() - start) * 1000)
                await self.on_result(job_id, task_id, "error", wall_ms, {"error": str(exc)})
        finally:
            # Tear down the RSS watchdog
            await ram_monitor.stop()
            self._current_task = None
            if self.backend:
                try:
                    self.backend.synchronize()
                    self.backend.cleanup()
                except Exception as e:
                    logger.debug(f"[{self.node_id}] Backend cleanup error: {e}")

    async def _load_weights(self, model: nn.Module, task_id: str, weight_url: str | None) -> None:
        if not weight_url:
            return
        safe_task_id = "".join(c for c in task_id if c.isalnum() or c in "-_")[:64]
        local_weights = Path("runtime/checkpoints") / f"{safe_task_id}_weights.pt"
        local_weights.parent.mkdir(parents=True, exist_ok=True)
        if not await self.s3.download_file(weight_url, str(local_weights)):
            raise ValueError(f"Failed to download weights from {weight_url}")
        try:
            state_dict = await asyncio.to_thread(torch.load, str(local_weights), weights_only=True)
            model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            logger.warning("[%s] Failed to load weights from %s: %s", self.node_id, weight_url, exc)

    async def _load_batch_source(self, task_id: str, batch_blob_url: str | None) -> dict[str, Any]:
        if not batch_blob_url:
            raise ValueError("A real batch_blob_url is required for training")
        if not is_allowed_gradient_url(batch_blob_url):
            logger.error("[SECURITY] Blocked batch download from disallowed URL: %s", batch_blob_url)
            raise ValueError(f"Unauthorized batch blob URL: {batch_blob_url}")
        parsed = urlparse(batch_blob_url)
        # A bare Windows path like C:\... parses with a single-letter "scheme"
        # (the drive letter); treat that the same as a local/file path.
        is_windows_drive_path = (
            len(parsed.scheme) == 1 and len(batch_blob_url) >= 3 and batch_blob_url[1:3] in (":\\", ":/")
        )
        if parsed.scheme in {"", "file"} or parsed.scheme in {"http", "https"} or is_windows_drive_path:
            text = await load_text_blob(batch_blob_url)
            if text is None:
                raise ValueError(f"Invalid or inaccessible batch file: {batch_blob_url}")
            return self._parse_batch_content(text)
        if parsed.scheme == "s3":
            safe_task_id = "".join(c for c in task_id if c.isalnum() or c in "-_")[:64]
            local_batch = Path("runtime/checkpoints") / f"{safe_task_id}_batch.data"
            if await self.s3.download_file(batch_blob_url, str(local_batch)):
                return self._parse_batch_content(local_batch.read_text(encoding="utf-8"))
            raise ValueError(f"Failed to download batch from {batch_blob_url}")
        raise ValueError(f"Unsupported batch_blob_url: {batch_blob_url}")

    def _parse_batch_content(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {"mode": "text", "content": content}
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"mode": "records", "records": parsed}
        return {"mode": "text", "content": content}

    def _build_language_model_batch(
        self,
        batch_source: dict[str, Any],
        batch_size: int,
        task_id: str,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = batch_source.get("content")
        if not text and batch_source.get("records"):
            text = "\n".join(json.dumps(item, sort_keys=True) for item in batch_source["records"])
        if not text:
            text = json.dumps(batch_source, sort_keys=True)
        tokens = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        if tokens.numel() < 17:
            tokens = torch.cat([tokens, tokens, tokens], dim=0)
        if seq_len is None:
            seq_len = min(32, max(8, tokens.numel() - 1))
        else:
            seq_len = min(int(seq_len), tokens.numel() - 1)
        start_seed = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest(), 16)
        windows = []
        targets = []
        max_start = max(1, tokens.numel() - seq_len - 1)
        for idx in range(batch_size):
            start = (start_seed + idx * seq_len) % max_start
            window = tokens[start : start + seq_len]
            target = tokens[start + 1 : start + seq_len + 1]
            windows.append(window)
            targets.append(target)
        return torch.stack(windows), torch.stack(targets)

    def _build_toy_batch(
        self, batch_source: dict[str, Any], batch_size: int, task_id: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        records = batch_source.get("records")
        if isinstance(records, list) and records:
            features = []
            labels = []
            for record in records[:batch_size]:
                if isinstance(record, dict):
                    features.append(record.get("features", [0.0] * 10))
                    labels.append(record.get("labels", [0.0] * 10))
            if features and labels:
                return torch.tensor(features, dtype=torch.float32), torch.tensor(
                    labels, dtype=torch.float32
                )
        text = batch_source.get("content") or json.dumps(batch_source, sort_keys=True)
        digest = hashlib.sha256(f"{task_id}:{text}".encode()).digest()
        values = torch.tensor(list(digest)[:10], dtype=torch.float32) / 255.0
        x = values.repeat(batch_size, 1)
        y = torch.roll(x, shifts=1, dims=1)
        return x, y

    def _compute_loss(
        self, model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        inputs, targets = batch
        try:
            device = next(model.parameters()).device
            inputs = inputs.to(device)
            targets = targets.to(device)
        except StopIteration:
            pass
        outputs = model(inputs)
        if self._is_language_model(model):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            elif hasattr(outputs, "logits"):
                # transformers ModelOutput (external/custom-code architectures).
                outputs = outputs.logits
            return functional.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
            )
        return functional.mse_loss(outputs, targets)

    def _is_language_model(self, model: nn.Module) -> bool:
        base_model = getattr(model, "_orig_mod", model)
        if isinstance(base_model, DistribAIModelWrapper):
            return True
        try:
            from transformers import PreTrainedModel
        except ImportError:
            return False
        return isinstance(base_model, PreTrainedModel)

    def _collect_gradients(self, model: nn.Module) -> tuple[dict[str, Any], float]:
        tensor_gradients: dict[str, torch.Tensor] = {}
        gradients: dict[str, Any] = {}
        total_norm = 0.0
        for name, value in model.named_parameters():
            if value.grad is None:
                continue
            if torch.isnan(value.grad).any() or torch.isinf(value.grad).any():
                raise ValueError(f"NaN/Inf detected in gradient {name}")
            # Low-precision external architectures (bfloat16/float16) produce
            # grads numpy can't convert directly; upcast before compression,
            # JSON serialization, and blob storage all downstream of this.
            grad_cpu = value.grad.detach().to(torch.float32).cpu()
            tensor_gradients[name] = grad_cpu
            gradients[name] = grad_cpu.tolist()
            total_norm += torch.norm(grad_cpu).item()
        if (
            tensor_gradients
            and os.getenv("DISTRIBAI_DISABLE_GRADIENT_COMPRESSION", "false").lower() != "true"
        ):
            compressed = self.gradient_compressor.compress(tensor_gradients)
            gradients = {
                "compression": "dgc",
                "raw_fallback": gradients,
                "compressed": self._jsonify_compressed_gradients(compressed),
            }
        return gradients, total_norm**0.5

    def _jsonify_compressed_gradients(self, compressed: dict[str, dict]) -> dict[str, dict]:
        payload: dict[str, dict] = {}
        for name, data in compressed.items():
            payload[name] = {
                "indices": data["indices"],
                "values": data["values"],
                "method": data["method"],
                "shape": list(data["shape"]),
            }
        return payload

    def _write_gradients(self, task_id: str, gradients: dict[str, Any]) -> Path:
        safe_task_id = "".join(c for c in task_id if c.isalnum() or c in "-_")[:64]
        path = Path("runtime/checkpoints") / f"{safe_task_id}_grads.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(gradients), encoding="utf-8")
        return path
