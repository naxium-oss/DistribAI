"""Docker isolation backend: ephemeral containers with tight defaults.

Chosen when ``docker`` is on PATH and ``docker info`` succeeds. The
factory falls through to nsjail/subprocess when the daemon is down.

Notable flags: ``--rm``, ``--read-only``, tmpfs ``/tmp``, RO task bind,
RW ``output/``, memory+swap caps, ``--cpus``, ``--pids-limit``, ``--cap-drop=ALL``, ``no-new-privileges``, network policy mapping,
non-root ``--user``. GPU passthrough only when ``DISTRIBAI_SANDBOX_GPU=1``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .base import NetworkPolicy, SandboxBackend, SandboxResult

logger = logging.getLogger(__name__)


DEFAULT_IMAGE = "python:3.11-slim"


def _docker_path_for_bind(path: Path) -> str:
    """Normalize a host path for ``docker -v`` bind mounts.

    Windows Docker Desktop wants forward slashes with the drive letter;
    POSIX can use ``str(path)`` unchanged.
    """
    if os.name == "nt":
        return str(path).replace("\\", "/")
    return str(path)


class DockerSandbox(SandboxBackend):
    """Execute task scripts in a short-lived hardened container."""

    name = "docker"

    def __init__(
        self,
        *,
        image: str | None = None,
        docker_bin: str | None = None,
        gpu: bool | None = None,
        pids_limit: int = 64,
        cpus: float = 2.0,
    ) -> None:
        self.image = image or os.getenv("DISTRIBAI_SANDBOX_IMAGE", DEFAULT_IMAGE)
        self.docker_bin = docker_bin or shutil.which("docker") or "docker"
        if gpu is None:
            gpu = os.getenv("DISTRIBAI_SANDBOX_GPU") == "1"
        self.gpu = bool(gpu)
        self.pids_limit = pids_limit
        self.cpus = cpus

    def is_available(self) -> bool:
        """Both the CLI and a reachable daemon are required."""
        if not shutil.which(self.docker_bin):
            return False
        try:
            cp = subprocess.run(
                [self.docker_bin, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return cp.returncode == 0

    def _network_args(self, network: NetworkPolicy) -> list[str]:
        if network is NetworkPolicy.NONE:
            return ["--network=none"]
        # RESTRICTED + OPEN both share the default bridge today.
        # Per-org egress ACLs are queued for a future release.
        return ["--network=bridge"]

    def _build_argv(
        self,
        *,
        container_name: str,
        task_dir: Path,
        env: dict[str, str],
        max_memory_mb: int,
        max_cpu_time_sec: int,
        network: NetworkPolicy,
    ) -> list[str]:
        output_dir = task_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        bind_task = _docker_path_for_bind(task_dir)
        bind_output = _docker_path_for_bind(output_dir)

        argv: list[str] = [
            self.docker_bin,
            "run",
            "--rm",
            "--name",
            container_name,
            "--read-only",
            "--tmpfs",
            "/tmp:size=1g,mode=1777",
            "-v",
            f"{bind_task}:/workspace:ro",
            "-v",
            f"{bind_output}:/workspace/output:rw",
            "--workdir",
            "/workspace",
            f"--memory={max_memory_mb}m",
            f"--memory-swap={max_memory_mb}m",
            f"--cpus={self.cpus}",
            f"--pids-limit={self.pids_limit}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user=1000:1000",
        ]
        argv.extend(self._network_args(network))

        if self.gpu:
            argv.extend(["--gpus", "all"])

        for k, v in env.items():
            argv.extend(["-e", f"{k}={v}"])

        argv.append(self.image)
        argv.extend(["python", "run.py"])
        return argv

    async def _docker_kill(self, container_name: str) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_bin,
                "kill",
                container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10)
        except (TimeoutError, OSError):
            logger.warning("docker kill %s failed", container_name)

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

        container_name = f"distribai-{uuid.uuid4().hex[:12]}"
        argv = self._build_argv(
            container_name=container_name,
            task_dir=task_dir,
            env=env,
            max_memory_mb=max_memory_mb,
            max_cpu_time_sec=max_cpu_time_sec,
            network=network,
        )
        logger.debug("DockerSandbox launching %s", " ".join(argv[:10]) + " ...")

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
                stderr=f"docker binary not found: {exc}",
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
            await self._docker_kill(container_name)
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=10,
                )
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
                "image": self.image,
                "container_name": container_name,
                "network_policy": network.value,
                "gpu": self.gpu,
            },
        )
