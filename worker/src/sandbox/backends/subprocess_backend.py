"""Hardened-subprocess isolation backend (portable fallback).

Encapsulates the v1.1 ``preexec_fn`` + ``Popen`` path as a
:class:`SandboxBackend`. Contract is locked by
``tests/security/test_v1_1_hardening.py``.

Selected on Windows and other hosts without Docker/nsjail. Provides
defense-in-depth only — not a true sandbox — so production fleets
should prefer Docker or nsjail via the factory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .base import NetworkPolicy, SandboxBackend, SandboxResult

logger = logging.getLogger(__name__)


def _build_preexec_fn(
    max_memory_mb: int,
    max_cpu_time_sec: int,
    max_file_size_mb: int,
    max_processes: int,
    max_open_files: int,
):
    """Build a POSIX ``preexec_fn`` that applies rlimits in the child.

    Limits hit the new process only. Returns ``None`` on Windows (do not
    pass that to ``preexec_fn``).
    """
    try:
        import resource as _resource
    except ImportError:  # Windows
        return None

    def _apply_limits() -> None:
        try:
            _resource.setrlimit(
                _resource.RLIMIT_AS,
                (max_memory_mb * 1024 * 1024, max_memory_mb * 1024 * 1024),
            )
            _resource.setrlimit(
                _resource.RLIMIT_CPU,
                (max_cpu_time_sec, max_cpu_time_sec + 60),
            )
            _resource.setrlimit(
                _resource.RLIMIT_FSIZE,
                (max_file_size_mb * 1024 * 1024, max_file_size_mb * 1024 * 1024),
            )
            _resource.setrlimit(_resource.RLIMIT_NPROC, (max_processes, max_processes))
            _resource.setrlimit(_resource.RLIMIT_NOFILE, (max_open_files, max_open_files))
            _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))
            try:
                import ctypes

                _PR_SET_NO_NEW_PRIVS = 38  # noqa: N806 -- kernel constant from <linux/prctl.h>
                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
            except (OSError, AttributeError):
                pass
        except (ValueError, OSError):
            # Setting rlimits can fail on some systems / container configs.
            # Swallow rather than abort the launch.
            pass

    return _apply_limits


class SubprocessSandbox(SandboxBackend):
    """v1.1 hardened-subprocess backend, lifted under the new interface."""

    name = "subprocess"

    def __init__(
        self,
        *,
        max_file_size_mb: int = 8 * 1024,
        max_processes: int = 64,
        max_open_files: int = 4096,
    ) -> None:
        self.max_file_size_mb = max_file_size_mb
        self.max_processes = max_processes
        self.max_open_files = max_open_files

    def is_available(self) -> bool:  # always available
        return True

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
        script_path = task_dir / "run.py"
        if not script_path.exists():
            return SandboxResult(
                return_code=-1,
                stdout="",
                stderr=f"run.py missing under {task_dir}",
                elapsed_seconds=0.0,
                timed_out=False,
                backend_used=self.name,
            )

        if network is not NetworkPolicy.OPEN:
            # SubprocessSandbox cannot enforce egress isolation; warn so
            # operators do not assume Docker-grade containment.
            logger.warning(
                "SubprocessSandbox cannot enforce NetworkPolicy.%s; "
                "install Docker or nsjail for true network isolation.",
                network.name,
            )

        preexec_fn = None
        if os.name == "posix":
            preexec_fn = _build_preexec_fn(
                max_memory_mb=max_memory_mb,
                max_cpu_time_sec=max_cpu_time_sec,
                max_file_size_mb=self.max_file_size_mb,
                max_processes=self.max_processes,
                max_open_files=self.max_open_files,
            )

        # asyncio.create_subprocess_exec does not accept preexec_fn on
        # all platforms; use the synchronous Popen behind a thread for
        # POSIX so we keep RLIMIT enforcement.
        creation_flags = 0
        if os.name == "nt":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        start = time.monotonic()
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(task_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=preexec_fn,
            close_fds=True,
            creationflags=creation_flags,
        )
        if on_process_started is not None:
            on_process_started(process)

        loop = asyncio.get_running_loop()
        try:
            stdout, stderr = await asyncio.wait_for(
                loop.run_in_executor(None, process.communicate),
                timeout=max_runtime_seconds,
            )
            timed_out = False
        except TimeoutError:
            process.kill()
            try:
                stdout, stderr = await loop.run_in_executor(None, process.communicate)
            except Exception:
                stdout, stderr = "", ""
            timed_out = True
        elapsed = time.monotonic() - start

        return SandboxResult(
            return_code=process.returncode if process.returncode is not None else -1,
            stdout=(stdout or "")[-10_000:],
            stderr=(stderr or "")[-10_000:],
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            backend_used=self.name,
            metadata={"network_policy": network.value},
        )
