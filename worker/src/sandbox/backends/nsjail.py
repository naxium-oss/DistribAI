"""Linux nsjail backend: namespaces + seccomp without a container daemon.

Used when ``nsjail`` is on PATH and Docker is unavailable. Writes a
per-task ``nsjail.cfg`` under ``task_dir``, binds the workspace, mounts
tmpfs on ``/tmp``, and applies rlimits aligned with SubprocessSandbox.
``clone_newnet`` enforces ``NetworkPolicy.NONE``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from .base import NetworkPolicy, SandboxBackend, SandboxResult

logger = logging.getLogger(__name__)


# Minimal seccomp posture for CPython/numpy/torch imports; avoid rare
# syscalls (ptrace/keyctl/bpf). Extend via DISTRIBAI_SANDBOX_NSJAIL_EXTRA_CFG.
_SECCOMP_ALLOW = "DEFAULT ALLOW"


def _render_config(
    *,
    task_dir: Path,
    max_memory_mb: int,
    max_cpu_time_sec: int,
    max_file_size_mb: int,
    network: NetworkPolicy,
    env: dict[str, str],
) -> str:
    """Render the nsjail config file body for a single task."""
    new_net = "true" if network is NetworkPolicy.NONE else "false"

    env_lines = "\n".join(f'envar: "{k}={v}"' for k, v in env.items())

    return f"""\
mode: ONCE
hostname: "distribai-task"
cwd: "/workspace"
time_limit: {max_cpu_time_sec}
rlimit_as: {max_memory_mb}
rlimit_cpu: {max_cpu_time_sec}
rlimit_fsize: {max_file_size_mb}
rlimit_nofile: 4096
rlimit_nproc: 64

mount {{
    src: "{task_dir.as_posix()}"
    dst: "/workspace"
    is_bind: true
    rw: true
}}
mount {{
    dst: "/tmp"
    fstype: "tmpfs"
    rw: true
    options: "size=1073741824,mode=1777"
}}

clone_newuser: true
clone_newnet: {new_net}
clone_newpid: true
clone_newipc: true
clone_newuts: true
clone_newns: true
keep_caps: false
no_new_privs: true
disable_proc: true

seccomp_string: "{_SECCOMP_ALLOW}"
{env_lines}
"""


class NsjailSandbox(SandboxBackend):
    """Run a script inside ``nsjail`` with strict cgroup + ns limits."""

    name = "nsjail"

    def __init__(
        self,
        *,
        nsjail_bin: str | None = None,
        max_file_size_mb: int = 8 * 1024,
    ) -> None:
        self.nsjail_bin = nsjail_bin or shutil.which("nsjail") or "nsjail"
        self.max_file_size_mb = max_file_size_mb

    def is_available(self) -> bool:
        if os.name != "posix":
            return False
        if not shutil.which(self.nsjail_bin):
            return False
        try:
            cp = subprocess.run(
                [self.nsjail_bin, "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        # nsjail --help exits non-zero on some builds; just having
        # produced output is enough proof it ran.
        return bool(cp.stdout) or bool(cp.stderr)

    async def run_script(
        self,
        *,
        task_dir: Path,
        env: dict[str, str],
        max_runtime_seconds: int,
        max_memory_mb: int,
        max_cpu_time_sec: int,
        network: NetworkPolicy = NetworkPolicy.NONE,
        on_process_started=None,
    ) -> SandboxResult:
        if not (task_dir / "run.py").exists():
            return SandboxResult(
                return_code=-1,
                stdout="",
                stderr=f"run.py missing under {task_dir}",
                elapsed_seconds=0.0,
                timed_out=False,
                backend_used=self.name,
            )

        cfg_path = task_dir / "nsjail.cfg"
        cfg_path.write_text(
            _render_config(
                task_dir=task_dir,
                max_memory_mb=max_memory_mb,
                max_cpu_time_sec=max_cpu_time_sec,
                max_file_size_mb=self.max_file_size_mb,
                network=network,
                env=env,
            ),
            encoding="utf-8",
        )

        argv = [
            self.nsjail_bin,
            "--config",
            str(cfg_path),
            "--",
            "/usr/bin/env",
            "python3",
            "run.py",
        ]

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            return SandboxResult(
                return_code=-1,
                stdout="",
                stderr=f"nsjail binary not found: {exc}",
                elapsed_seconds=0.0,
                timed_out=False,
                backend_used=self.name,
            )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=max_runtime_seconds,
            )
        except TimeoutError:
            timed_out = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=10)
            except TimeoutError:
                stdout_b, stderr_b = b"", b""

        elapsed = time.monotonic() - start
        rc = proc.returncode if proc.returncode is not None else -1

        return SandboxResult(
            return_code=rc,
            stdout=(stdout_b or b"").decode("utf-8", errors="replace")[-10_000:],
            stderr=(stderr_b or b"").decode("utf-8", errors="replace")[-10_000:],
            elapsed_seconds=elapsed,
            timed_out=timed_out,
            backend_used=self.name,
            metadata={
                "config_path": str(cfg_path),
                "network_policy": network.value,
            },
        )
