"""Opt-in native C++ sandbox backend (``DISTRIBAI_SANDBOX_BACKEND=native-cpp``).

Routes **native-compatible training jobs** (``hyperparams["native_train"]``
truthy) through the C++ port's sandboxed trainer
(``build/cpp_port/distribai_dualrun --sandbox``): fork + exact rlimits +
user/PID namespaces + POD pipe handoff. Behavioral parity with the Python
path is locked by ``tools/cpp_port/tests/test_golden.cpp`` and the dual-run
gate (``runtime/baselines/dualrun_diffs.json``); the result envelope matches
the baseline sandbox contract (status/n_params/steps_per_s/final_loss/...).

All other payloads delegate to :class:`SubprocessSandbox`, general script
execution stays on the Python path. ``SandboxResult.backend_used`` is
``"native-cpp"`` only for jobs actually run natively.

The binary is *required* (probed by the factory via :meth:`is_available`)
and its location can be overridden with ``DISTRIBAI_NATIVE_TRAIN_BIN``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import NetworkPolicy, SandboxResult
from .subprocess_backend import SubprocessSandbox

logger = logging.getLogger(__name__)

_DEFAULT_BIN = Path(__file__).resolve().parents[5] / "build" / "cpp_port" / "distribai_dualrun"


def _resolve_bin() -> Path:
    override = os.getenv("DISTRIBAI_NATIVE_TRAIN_BIN")
    return Path(override) if override else _DEFAULT_BIN


def _truthy(value: Any) -> bool:
    return bool(value) and value not in ("0", "false", "False", "no", "off")


class NativeCppSandbox(SubprocessSandbox):
    """C++ hot-path backend; opt-in, parity-locked, subprocess fallback."""

    name = "native-cpp"

    def __init__(self, *, binary: Path | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._binary = binary or _resolve_bin()

    def is_available(self) -> bool:
        return self._binary.is_file() and os.access(self._binary, os.X_OK)

    async def run_script(
        self,
        *,
        task_dir: Path,
        env: dict[str, str],
        max_runtime_seconds: int,
        max_memory_mb: int,
        max_cpu_time_sec: int,
        network: NetworkPolicy = NetworkPolicy.NONE,
        on_process_started: Callable[[subprocess.Popen], None] | None = None,
    ) -> SandboxResult:
        """Run the task; native trainer when the payload opts in."""
        hyperparams: dict[str, Any] = {}
        try:
            hp_path = task_dir / "hyperparams.json"
            if hp_path.is_file():
                hyperparams = json.loads(hp_path.read_text() or "{}")
        except (OSError, json.JSONDecodeError):
            hyperparams = {}

        if not _truthy(hyperparams.get("native_train")):
            logger.debug("native-cpp backend: payload not native_train; using subprocess")
            result = await super().run_script(
                task_dir=task_dir,
                env=env,
                max_runtime_seconds=max_runtime_seconds,
                max_memory_mb=max_memory_mb,
                max_cpu_time_sec=max_cpu_time_sec,
                network=network,
                on_process_started=on_process_started,
            )
            # Audit trail: the delegated run executed on the subprocess path,
            # not this backend, report the true executor.
            if result.backend_used == self.name:
                result.backend_used = "subprocess"
            return result

        seed = int(hyperparams.get("seed", 42)) & 0x7FFFFFFF
        steps = max(1, min(int(hyperparams.get("steps", 200)), 100_000))
        mem_mb = max(8, min(int(hyperparams.get("mem_mb", max_memory_mb)), max_memory_mb))
        cpu_sec = max(1, min(int(hyperparams.get("cpu_sec", max_cpu_time_sec)), max_cpu_time_sec))

        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            str(self._binary),
            "--seed",
            str(seed),
            "--steps",
            str(steps),
            "--sandbox",
            "--mem-mb",
            str(mem_mb),
            "--cpu-sec",
            str(cpu_sec),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=max_runtime_seconds)
        except TimeoutError:  # asyncio.TimeoutError is an alias on py3.11+
            proc.kill()
            await proc.wait()
            return SandboxResult(
                return_code=-9,
                stdout="",
                stderr="native-cpp trainer timed out",
                elapsed_seconds=time.monotonic() - started,
                timed_out=True,
                backend_used=self.name,
            )
        elapsed = time.monotonic() - started
        out_text = stdout.decode(errors="replace")

        # Exit code: honor the child's; a zero exit with a malformed envelope
        # is still a failure for this backend.
        envelope_ok = '"status": "ok"' in out_text or '"status":"ok"' in out_text
        rc = proc.returncode if proc.returncode is not None else 1
        if rc == 0 and not envelope_ok:
            rc = 1
        return SandboxResult(
            return_code=rc,
            stdout=out_text,
            stderr=stderr.decode(errors="replace"),
            elapsed_seconds=elapsed,
            timed_out=False,
            backend_used=self.name,
        )
