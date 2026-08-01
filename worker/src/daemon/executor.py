"""
Per-task training executor used by the worker daemon.

The training loop honors job-level hyperparameters (learning rate, weight
decay, gradient clipping/accumulation, warmup + LR schedules, mixed
precision, tokenizer choice), rebuilds fresh data windows every step,
periodically checkpoints model+optimizer state for crash recovery, and
recovers from OOM by genuinely halving the batch and retrying the step.
"""

import asyncio
import gc
import hashlib
import json
import logging
import math
import os
import platform
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as functional

from services_python.blob_loader import load_text_blob
from services_python.blob_url_policy import is_allowed_gradient_url

from .oom_guard import OOMGuard, OOMStrategy, is_oom_error
from .optimizers import build_optimizer


def _resolve_default_optimizer() -> str:
    raw = os.getenv("DISTRIBAI_DEFAULT_OPTIMIZER", "auon").strip().lower()
    return raw or "auon"


def _resolve_job_optimizer(hyperparams: dict[str, Any]) -> str:
    override = hyperparams.get("optimizer")
    if override:
        return str(override).strip().lower()
    return _resolve_default_optimizer()


@dataclass(frozen=True)
class TrainingSettings:
    """Validated per-job training knobs with safe fallbacks.

    Every field comes from job hyperparameters but is clamped to a sane
    range, so a malformed submission degrades to defaults instead of
    crashing the worker or letting a job request absurd values.
    """

    lr: float
    weight_decay: float
    grad_clip: float  # 0 disables clipping
    grad_accum_steps: int
    warmup_steps: int
    lr_schedule: str  # "constant" | "cosine" | "linear"
    mixed_precision: str  # "auto" | "off" | "fp16" | "bf16"
    seed: int | None
    checkpoint_every: int  # 0 disables mid-run training-state checkpoints


def _clamped_float(
    raw: Any, default: float, minimum: float, maximum: float, name: str
) -> float:
    """Parse a float hyperparameter, clamping into [minimum, maximum]."""
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Ignoring non-numeric hyperparameter %s=%r; using %s", name, raw, default
        )
        return default
    if not math.isfinite(value):
        return default
    return min(maximum, max(minimum, value))


def _clamped_int(raw: Any, default: int, minimum: int, maximum: int, name: str) -> int:
    """Parse an int hyperparameter, clamping into [minimum, maximum]."""
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logging.getLogger(__name__).warning(
            "Ignoring non-integer hyperparameter %s=%r; using %s", name, raw, default
        )
        return default
    return min(maximum, max(minimum, value))


def resolve_training_settings(
    hyperparams: dict[str, Any] | None, *, language_model: bool
) -> TrainingSettings:
    """Extract, validate, and clamp the training knobs a job may set.

    Args:
        hyperparams: Raw job hyperparameters (untrusted; values clamped).
        language_model: Selects the LM defaults (lower LR, gradient clipping)
            versus the toy-regression defaults that preserve historic
            behavior for non-LM payloads.

    Returns:
        Immutable :class:`TrainingSettings`.
    """
    hp = hyperparams or {}
    default_lr = 0.001 if language_model else 0.01
    schedule = str(hp.get("lr_schedule", "constant")).strip().lower()
    if schedule not in {"constant", "cosine", "linear"}:
        logging.getLogger(__name__).warning(
            "Unknown lr_schedule %r; using constant", schedule
        )
        schedule = "constant"
    precision = str(hp.get("mixed_precision", "auto")).strip().lower()
    if precision not in {"auto", "off", "fp16", "bf16"}:
        logging.getLogger(__name__).warning(
            "Unknown mixed_precision %r; using auto", precision
        )
        precision = "auto"
    seed_raw = hp.get("seed")
    seed: int | None = None
    if seed_raw is not None:
        try:
            seed = int(seed_raw) & 0x7FFFFFFF
        except (TypeError, ValueError):
            seed = None
    return TrainingSettings(
        lr=_clamped_float(hp.get("lr"), default_lr, 1e-6, 1.0, "lr"),
        weight_decay=_clamped_float(hp.get("weight_decay"), 0.0, 0.0, 1.0, "weight_decay"),
        grad_clip=_clamped_float(
            hp.get("grad_clip"), 1.0 if language_model else 0.0, 0.0, 1e4, "grad_clip"
        ),
        grad_accum_steps=_clamped_int(
            hp.get("grad_accum_steps"), 1, 1, 64, "grad_accum_steps"
        ),
        warmup_steps=_clamped_int(hp.get("warmup_steps"), 0, 0, 100_000, "warmup_steps"),
        lr_schedule=schedule,
        mixed_precision=precision,
        seed=seed,
        checkpoint_every=_clamped_int(
            hp.get("checkpoint_every_steps"), 0, 0, 100_000, "checkpoint_every_steps"
        ),
    )


def build_lr_lambda(settings: TrainingSettings, total_steps: int) -> Callable[[int], float]:
    """LR multiplier by 0-based optimizer step: linear warmup then decay.

    Cosine decays to ~0 at the final step, linear keeps a 5% floor so the
    last updates still move, and constant holds 1.0 after warmup.
    """
    warmup = min(settings.warmup_steps, max(0, total_steps - 1))
    decay_span = max(1, total_steps - warmup)

    def lr_lambda(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if settings.lr_schedule == "cosine":
            progress = min(1.0, (step - warmup) / decay_span)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if settings.lr_schedule == "linear":
            progress = min(1.0, (step - warmup) / decay_span)
            return max(0.05, 1.0 - progress)
        return 1.0

    return lr_lambda


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
    from ..compute.external_arch import load_external_architecture, looks_like_external_model_ref
    from ..compute.hf_integration import get_tokenizer_from_hf
    from .gradient_compression import DeepGradientCompression
    from .s3_util import S3Manager
except ImportError:
    try:
        from compute import ComputeBackend, detect_backend
        from compute.distribai_models import CustomModelBuilder, DistribAIModelWrapper, get_model
        from compute.external_arch import load_external_architecture, looks_like_external_model_ref
        from compute.hf_integration import get_tokenizer_from_hf
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
        from compute.external_arch import load_external_architecture, looks_like_external_model_ref
        from compute.hf_integration import get_tokenizer_from_hf
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
        vocab_size: int = 256,
    ) -> nn.Module:
        """Build the model a task requests: native family, profile, or external ref.

        Args:
            model_name: Named profile, legacy alias, or external reference.
            architecture_config: Validated declarative family config (wins
                over every other source when present).
            hyperparams: Job hyperparameters (external-model refs, dtype,
                trust flags).
            vocab_size: Vocabulary width for native families — 256 for the
                default byte-level pipeline, or the tokenizer's size when the
                job supplies one.
        """
        model_name_lower = model_name.lower()
        hp = hyperparams or {}
        if architecture_config is not None:
            logger.info(
                "Creating uploaded DistribAI architecture family %s",
                architecture_config.get("family", architecture_config.get("architecture")),
            )
            return get_model(
                model_name_lower or "uploaded-architecture",
                vocab_size=vocab_size,
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
            return get_model(model_name_lower, vocab_size=vocab_size)
        if model_name_lower == "custom":
            logger.info("Creating explicitly requested custom DistribAI model")
            return CustomModelBuilder.create_custom_model(vocab_size=vocab_size)
        if model_name_lower in {"tiny", "small", "medium"}:
            return get_model(model_name_lower, vocab_size=vocab_size)
        if model_name_lower == "toy" and os.getenv("DISTRIBAI_ALLOW_TEST_MODELS") == "1":
            logger.warning("Creating test-only ToyModel because DISTRIBAI_ALLOW_TEST_MODELS=1")
            return ToyModel()
        raise ValueError(
            f"Unknown model profile {model_name!r}; available profiles: "
            f"{sorted(DistribAIModelWrapper.MODEL_CONFIGS)} plus custom, "
            "or an external Hub/local architecture reference"
        )

    def _resolve_tokenizer(self, hyperparams: dict[str, Any] | None) -> Any | None:
        """Load the job's optional Hugging Face tokenizer.

        Returns None (byte-level fallback) when unset or when loading fails —
        a bad tokenizer name must degrade, not kill the task.
        """
        name = (hyperparams or {}).get("tokenizer")
        if not name or not isinstance(name, str):
            return None
        try:
            tokenizer = get_tokenizer_from_hf(name.strip())
            logger.info(
                "[%s] Using HF tokenizer %s (vocab_size=%s)",
                self.node_id,
                name,
                getattr(tokenizer, "vocab_size", "?"),
            )
            return tokenizer
        except Exception as exc:
            logger.warning(
                "[%s] Failed to load tokenizer %r (%s); using byte-level fallback",
                self.node_id,
                name,
                exc,
            )
            return None

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
            tokenizer = self._resolve_tokenizer(hyperparams)
            vocab_size = 256
            if tokenizer is not None:
                try:
                    vocab_size = max(2, int(len(tokenizer)))
                except TypeError:
                    vocab_size = max(2, int(getattr(tokenizer, "vocab_size", 256) or 256))
            try:
                with self.oom_guard.guard("model_creation"):
                    if (
                        architecture_config is None
                        and tokenizer is None
                        and not (
                            (hyperparams or {}).get("external_model")
                            or (hyperparams or {}).get("hf_model_id")
                            or (hyperparams or {}).get("hf_repo")
                        )
                    ):
                        # Keep the one-arg call path for callers that patch
                        # or subclass _create_model(model_name) only.
                        model = self._create_model(model_name)
                    else:
                        model = self._create_model(
                            model_name,
                            architecture_config=architecture_config,
                            hyperparams=hyperparams,
                            vocab_size=vocab_size,
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
            is_language_model = self._is_language_model(model)
            settings = resolve_training_settings(hyperparams, language_model=is_language_model)
            if settings.seed is not None:
                torch.manual_seed(settings.seed)
            configured_seq_len: int | None = None
            if is_language_model:
                model_config = getattr(model, "config", None)
                configured_seq_len = getattr(model_config, "seq_len", None) or getattr(
                    model_config, "max_position_embeddings", None
                )
                train_batch = self._build_language_model_batch(
                    batch_source,
                    batch_size,
                    task_id,
                    seq_len=configured_seq_len,
                    tokenizer=tokenizer,
                )
            else:
                train_batch = self._build_toy_batch(batch_source, batch_size, task_id)
            optimizer = build_optimizer(
                optimizer_name,
                model.parameters(),
                lr=settings.lr,
                weight_decay=settings.weight_decay,
            )
            lr_multiplier = build_lr_lambda(settings, steps)

            # Mixed precision: CUDA-only. "auto" prefers bf16 (no loss scaling
            # needed) and falls back to fp16 with a GradScaler.
            device_type = "cpu"
            try:
                device_type = next(model.parameters()).device.type
            except StopIteration:
                logger.debug("[%s] Model has no parameters; assuming CPU", self.node_id)
            amp_dtype: torch.dtype | None = None
            if settings.mixed_precision != "off" and device_type == "cuda":
                prefer_bf16 = settings.mixed_precision == "bf16" or (
                    settings.mixed_precision == "auto" and torch.cuda.is_bf16_supported()
                )
                amp_dtype = torch.bfloat16 if prefer_bf16 else torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

            # Crash recovery: resume from the task's training-state checkpoint
            # when mid-run checkpointing is enabled and a state file exists.
            start_step = 1
            if settings.checkpoint_every > 0:
                resumed_step = self._load_training_state(task_id, model, optimizer)
                if resumed_step > 0:
                    start_step = min(resumed_step + 1, steps + 1)
                    logger.info(
                        "[%s] Resumed task %s from checkpointed step %s",
                        self.node_id,
                        task_id,
                        resumed_step,
                    )

            initial_loss = 0.0
            loss_val = 0.0
            current_batch_size = batch_size
            oom_budget = max(0, self.oom_guard.max_retries)
            try:
                step = start_step
                while step <= steps:
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
                    # Warmup/decay schedule applied directly so resume never
                    # depends on scheduler object state.
                    step_lr = settings.lr * lr_multiplier(step - 1)
                    for group in optimizer.param_groups:
                        group["lr"] = step_lr
                    try:
                        optimizer.zero_grad()
                        accumulated_loss = 0.0
                        for micro in range(settings.grad_accum_steps):
                            if is_language_model:
                                # Fresh windows every micro-step: the offset
                                # advances with (step, micro) so the task
                                # sweeps the blob instead of memorizing one
                                # fixed batch.
                                micro_batch = self._build_language_model_batch(
                                    batch_source,
                                    current_batch_size,
                                    task_id,
                                    seq_len=configured_seq_len,
                                    tokenizer=tokenizer,
                                    sample_offset=(step - 1) * settings.grad_accum_steps + micro,
                                )
                            else:
                                micro_batch = train_batch
                            with torch.autocast(
                                device_type=device_type,
                                dtype=amp_dtype,
                                enabled=amp_dtype is not None,
                            ):
                                loss = (
                                    self._compute_loss(model, micro_batch)
                                    / settings.grad_accum_steps
                                )
                            if scaler.is_enabled():
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                            accumulated_loss += float(loss.detach().item())
                        if settings.grad_clip > 0:
                            if scaler.is_enabled():
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                    except (RuntimeError, MemoryError) as step_exc:
                        # Real OOM recovery: halve the batch and retry this
                        # same step while the retry budget lasts.
                        if (
                            not is_oom_error(step_exc)
                            or current_batch_size <= 1
                            or oom_budget <= 0
                        ):
                            raise
                        self.oom_guard.handle_oom(step_exc, "training_step")
                        oom_budget -= 1
                        current_batch_size = max(1, current_batch_size // 2)
                        optimizer.zero_grad(set_to_none=True)
                        if not is_language_model:
                            train_batch = self._build_toy_batch(
                                batch_source, current_batch_size, task_id
                            )
                        logger.warning(
                            "[%s] OOM at step %s; retrying with batch_size=%s (%s retries left)",
                            self.node_id,
                            step,
                            current_batch_size,
                            oom_budget,
                        )
                        continue
                    if self.backend:
                        self.backend.synchronize()
                    loss_val = accumulated_loss
                    if step == start_step:
                        initial_loss = loss_val
                    if step % progress_every == 0 or step == steps:
                        await self.on_progress(job_id, task_id, step, loss_val)
                    if (
                        settings.checkpoint_every > 0
                        and step < steps
                        and step % settings.checkpoint_every == 0
                    ):
                        self._save_training_state(task_id, model, optimizer, step, loss_val)
                    await asyncio.sleep(0)
                    step += 1
                gradients, gradient_norm = self._collect_gradients(model)
                gradients_path = self._write_gradients(task_id, gradients)
                gradient_key = f"gradients/{job_id}/{gradients_path.name}"
                gradient_url = await self.s3.upload_file(str(gradients_path), gradient_key)
                self._clear_training_state(task_id)
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
                        "lr": settings.lr,
                        "effective_batch_size": current_batch_size * settings.grad_accum_steps,
                        "mixed_precision": str(amp_dtype).replace("torch.", "")
                        if amp_dtype is not None
                        else "off",
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

    def _encode_text(self, text: str, tokenizer: Any | None) -> torch.Tensor:
        """Token ids for training text: HF tokenizer when set, UTF-8 bytes otherwise."""
        if tokenizer is not None:
            try:
                token_ids = tokenizer.encode(text, add_special_tokens=False)
                if token_ids:
                    return torch.tensor(token_ids, dtype=torch.long)
            except Exception as exc:
                logger.warning(
                    "[%s] Tokenizer encode failed (%s); using byte-level fallback",
                    self.node_id,
                    exc,
                )
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)

    def _build_language_model_batch(
        self,
        batch_source: dict[str, Any],
        batch_size: int,
        task_id: str,
        seq_len: int | None = None,
        tokenizer: Any | None = None,
        sample_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-token windows from the batch blob.

        Args:
            batch_source: Parsed blob content ("content"/"records"/raw dict).
            batch_size: Number of windows to stack.
            task_id: Seeds window placement so replicas of the same task see
                the same data while different tasks see different slices.
            seq_len: Window length cap (model's context when known).
            tokenizer: Optional HF tokenizer; None keeps the byte-level path.
            sample_offset: Advances the deterministic window seed so each
                (step, micro-step) trains on fresh slices of the blob.
        """
        text = batch_source.get("content")
        if not text and batch_source.get("records"):
            text = "\n".join(json.dumps(item, sort_keys=True) for item in batch_source["records"])
        if not text:
            text = json.dumps(batch_source, sort_keys=True)
        tokens = self._encode_text(text, tokenizer)
        if tokens.numel() < 17:
            tokens = torch.cat([tokens, tokens, tokens], dim=0)
        if seq_len is None:
            seq_len = min(32, max(8, tokens.numel() - 1))
        else:
            seq_len = max(1, min(int(seq_len), tokens.numel() - 1))
        seed_text = task_id if sample_offset == 0 else f"{task_id}:{sample_offset}"
        start_seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest(), 16)
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
            logger.debug("[%s] Model has no parameters; batch stays on CPU", self.node_id)
        base_model = getattr(model, "_orig_mod", model)
        if isinstance(base_model, DistribAIModelWrapper):
            # Native wrapper owns the family-aware loss (adds MTP auxiliary
            # heads when the architecture declares extra horizons).
            return base_model.compute_loss(inputs, targets)
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

    def _training_state_path(self, task_id: str) -> Path:
        safe_task_id = "".join(c for c in task_id if c.isalnum() or c in "-_")[:64]
        return Path("runtime/checkpoints") / f"{safe_task_id}_train_state.pt"

    def _save_training_state(
        self,
        task_id: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
    ) -> None:
        """Atomically persist model+optimizer state for crash recovery.

        Written to a temp file first and swapped in with ``replace`` so a
        crash mid-write can never leave a truncated checkpoint behind.
        Failures are logged and swallowed: checkpointing is an optimization,
        not a correctness requirement for the current run.
        """
        path = self._training_state_path(task_id)
        temp_path = path.with_suffix(".pt.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": int(step),
                    "loss": float(loss),
                },
                str(temp_path),
            )
            temp_path.replace(path)
            logger.debug("[%s] Saved training state for %s at step %s", self.node_id, task_id, step)
        except (OSError, RuntimeError) as exc:
            logger.warning("[%s] Failed to save training state: %s", self.node_id, exc)
            temp_path.unlink(missing_ok=True)

    def _load_training_state(
        self, task_id: str, model: nn.Module, optimizer: torch.optim.Optimizer
    ) -> int:
        """Restore a previous run's model+optimizer state for this task.

        Returns:
            The completed step recorded in the checkpoint, or 0 when no
            usable state exists (fresh start).
        """
        path = self._training_state_path(task_id)
        if not path.exists():
            return 0
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            return max(0, int(state.get("step", 0)))
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            logger.warning(
                "[%s] Ignoring unusable training state for %s: %s", self.node_id, task_id, exc
            )
            return 0

    def _clear_training_state(self, task_id: str) -> None:
        """Drop the crash-recovery state once the task finished successfully."""
        try:
            self._training_state_path(task_id).unlink(missing_ok=True)
        except OSError as exc:
            logger.debug("[%s] Could not remove training state: %s", self.node_id, exc)
