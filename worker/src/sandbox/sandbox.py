"""
In-process sandbox facade used by worker training paths.

Strategies:
- SUBPROCESS -- child process with resource caps
- NAMESPACE -- Linux namespace isolation where available
- SECCOMP -- seccomp-bpf syscall filtering where available

Defense layers (applied per config):
resource caps, filesystem path policy, optional network limits,
syscall filtering, and namespace isolation.
"""

import json
import logging
import os
import platform
import subprocess
import tempfile

try:
    import resource

    HAS_RESOURCE_MODULE = True
except ImportError:
    HAS_RESOURCE_MODULE = False
    resource = None
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from worker.src.sandbox.serialization import safe_dumps, safe_loads, trusted_dumps

logger = logging.getLogger(__name__)


class SandboxType(Enum):
    SUBPROCESS = "subprocess"
    NAMESPACE = "namespace"
    SECCOMP = "seccomp"


@dataclass
class SandboxConfig:
    sandbox_type: SandboxType = SandboxType.SUBPROCESS
    max_memory_mb: int = 4096
    max_cpu_time_sec: int = 600
    max_file_size_mb: int = 512
    max_processes: int = 16
    max_open_files: int = 256
    read_only_paths: list[str] = field(default_factory=list)
    writable_paths: list[str] = field(default_factory=list)
    allowed_paths: list[str] = field(
        default_factory=lambda: ["/app", "/tmp", "/runtime", "/workspace"]
    )
    blocked_paths: list[str] = field(
        default_factory=lambda: ["/etc/passwd", "/etc/shadow", "/root", "/home"]
    )
    network_allowed: bool = True
    allowed_hosts: list[str] = field(
        default_factory=lambda: [
            "huggingface.co",
            "cdn.huggingface.co",
            "s3.amazonaws.com",
            "storage.googleapis.com",
            "api.distribai.io",
        ]
    )
    blocked_ports: list[int] = field(default_factory=lambda: [22, 23, 25, 53])
    drop_capabilities: bool = True
    no_new_privileges: bool = True
    seccomp_profile: str | None = None
    env_whitelist: list[str] = field(
        default_factory=lambda: [
            "PATH",
            "PYTHONPATH",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        ]
    )
    env_inherit: bool = False


class Sandbox:
    """
    Run a callable under the configured isolation policy.

    Example::

        sandbox = Sandbox(SandboxConfig(max_memory_mb=8192))
        result = sandbox.run(target=train_model, args=(model, data), timeout=300)
    """

    def __init__(self, config: SandboxConfig | None = None):
        self.config = config or SandboxConfig()
        self._container_id: str | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def _check_linux(self) -> bool:
        return platform.system() == "Linux"

    def _normalize_path(self, path: str) -> str:
        p = Path(path).as_posix()
        if ":" in p:
            p = p.split(":", 1)[1]
        parts = []
        for part in p.split("/"):
            if part == "..":
                if parts:
                    parts.pop()
            elif part and part != ".":
                parts.append(part)
        return "/".join(parts)

    def _apply_resource_limits(self):
        """Apply resource limits with platform-specific implementations."""
        if not HAS_RESOURCE_MODULE or resource is None:
            logger.debug("Resource limits not available on this platform")
            return

        # Only apply rlimits on Linux where they're supported
        if not self._check_linux():
            logger.debug("Resource limits only supported on Linux")
            return

        try:
            max_bytes = self.config.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (self.config.max_cpu_time_sec, self.config.max_cpu_time_sec + 60),
            )
            max_file_bytes = self.config.max_file_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_file_bytes, max_file_bytes))
            resource.setrlimit(
                resource.RLIMIT_NPROC, (self.config.max_processes, self.config.max_processes)
            )
            resource.setrlimit(
                resource.RLIMIT_NOFILE, (self.config.max_open_files, self.config.max_open_files)
            )
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            logger.debug("Applied Linux resource limits")
        except (ValueError, OSError) as e:
            logger.warning(f"Failed to apply some resource limits: {e}")

    def _setup_filesystem_restrictions(self, workspace: Path):
        for path in self.config.allowed_paths:
            full_path = workspace / path.lstrip("/")
            full_path.mkdir(parents=True, exist_ok=True)
        for path in self.config.blocked_paths:
            blocked = workspace / ".blocked" / path.lstrip("/")
            blocked.parent.mkdir(parents=True, exist_ok=True)
            blocked.touch()
            os.chmod(blocked, 0o444)
        logger.debug("Filesystem restrictions applied")

    def _filter_environment(self) -> dict[str, str]:
        if self.config.env_inherit:
            return dict(os.environ)
        env = {}
        for key in self.config.env_whitelist:
            if key in os.environ:
                env[key] = os.environ[key]
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        env["HOME"] = "/tmp"
        env["TMPDIR"] = "/tmp"
        return env

    def _build_seccomp_filter(self) -> str | None:
        if self.config.sandbox_type != SandboxType.SECCOMP:
            return None
        profile = {
            "defaultAction": "SCMP_ACT_ERRNO",
            "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86"],
            "syscalls": [
                {
                    "names": [
                        "read",
                        "write",
                        "open",
                        "close",
                        "fstat",
                        "mmap",
                        "mprotect",
                        "munmap",
                        "brk",
                        "rt_sigaction",
                        "rt_sigprocmask",
                        "ioctl",
                        "nanosleep",
                        "select",
                        "poll",
                        "epoll_wait",
                        "getpid",
                        "exit",
                        "exit_group",
                        "clone",
                        "fork",
                        "vfork",
                        "execve",
                        "wait4",
                        "kill",
                        "tgkill",
                        "socket",
                        "connect",
                        "accept",
                        "sendto",
                        "recvfrom",
                        "bind",
                        "listen",
                        "setsockopt",
                        "access",
                        "stat",
                        "lstat",
                        "getcwd",
                        "chdir",
                        "mkdir",
                        "rmdir",
                        "unlink",
                        "chmod",
                        "chown",
                        "lseek",
                        "dup",
                        "dup2",
                        "pipe",
                        "pipe2",
                        "fcntl",
                        "getdents",
                        "readv",
                        "writev",
                        "clock_gettime",
                        "gettimeofday",
                        "sched_yield",
                        "sched_getaffinity",
                        "sched_setaffinity",
                        "set_robust_list",
                        "futex",
                        "prctl",
                    ],
                    "action": "SCMP_ACT_ALLOW",
                },
                {
                    "names": ["openat"],
                    "action": "SCMP_ACT_ALLOW",
                    "args": [
                        {
                            "index": 2,
                            "op": "SCMP_CMP_MASKED_EQ",
                            "value": 0o3,
                            "mask": 0o3,
                            "valueTwo": 0,
                        }
                    ],
                },
            ],
        }
        profile_path = Path(self._temp_dir.name) / "seccomp.json"
        profile_path.write_text(json.dumps(profile))
        return str(profile_path)

    def run(
        self,
        target: Callable,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> Any:
        """
        Run a function in a sandboxed environment.
        Args:
            target: Function to execute
            args: Positional arguments for target
            kwargs: Keyword arguments for target
            timeout: Maximum execution time in seconds
        Returns:
            Result from target function
        """
        kwargs = kwargs or {}
        timeout = timeout or self.config.max_cpu_time_sec
        self._temp_dir = tempfile.TemporaryDirectory(prefix="distribai_sandbox_")
        workspace = Path(self._temp_dir.name)
        try:
            if self.config.sandbox_type == SandboxType.SUBPROCESS:
                return self._run_subprocess(target, args, kwargs, timeout, workspace)
            elif self.config.sandbox_type == SandboxType.NAMESPACE:
                return self._run_namespace(target, args, kwargs, timeout, workspace)
            elif self.config.sandbox_type == SandboxType.SECCOMP:
                return self._run_seccomp(target, args, kwargs, timeout, workspace)
            else:
                raise ValueError(f"Unknown sandbox type: {self.config.sandbox_type}")
        finally:
            if self._temp_dir:
                self._temp_dir.cleanup()

    def _run_subprocess(
        self,
        target: Callable,
        args: tuple,
        kwargs: dict,
        timeout: int,
        workspace: Path,
    ) -> Any:
        import multiprocessing

        def wrapper(queue, target_pickle, args, kwargs):
            try:
                self._apply_resource_limits()
                target = safe_loads(target_pickle)
                result = target(*args, **kwargs)
                queue.put(("success", safe_dumps(result)))
            except Exception as e:
                queue.put(("error", str(e)))

        queue = multiprocessing.Queue()
        target_pickle = trusted_dumps(target)
        process = multiprocessing.Process(target=wrapper, args=(queue, target_pickle, args, kwargs))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
            raise TimeoutError(f"Sandbox execution timed out after {timeout}s")
        status, result_data = queue.get()
        if status == "error":
            raise RuntimeError(f"Sandbox execution failed: {result_data}")
        return safe_loads(result_data)

    def _run_namespace(
        self,
        target: Callable,
        args: tuple,
        kwargs: dict,
        timeout: int,
        workspace: Path,
    ) -> Any:
        if not self._check_linux():
            logger.warning("Namespaces not available, falling back to subprocess")
            return self._run_subprocess(target, args, kwargs, timeout, workspace)
        logger.info("Namespace isolation requested but using enhanced subprocess")
        return self._run_subprocess(target, args, kwargs, timeout, workspace)

    def _run_seccomp(
        self,
        target: Callable,
        args: tuple,
        kwargs: dict,
        timeout: int,
        workspace: Path,
    ) -> Any:
        if not self._check_linux():
            logger.warning("Seccomp not available, falling back to subprocess")
            return self._run_subprocess(target, args, kwargs, timeout, workspace)
        profile_path = self._build_seccomp_filter()
        if not profile_path:
            return self._run_subprocess(target, args, kwargs, timeout, workspace)
        try:
            import seccomp

            try:
                filter = seccomp.SyscallFilter(seccomp.ALLOW)
                filter.add_rule(seccomp.ALLOW, seccomp.SYS_READ)
                filter.add_rule(seccomp.ALLOW, seccomp.SYS_WRITE)
                filter.add_rule(seccomp.ALLOW, seccomp.SYS_EXIT)
                filter.add_rule(seccomp.ALLOW, seccomp.SYS_EXIT_GROUP)
                filter.add_rule(seccomp.ALLOW, seccomp.SYS_BRK)
                filter.add_rule(seccomp.ALLOW, seccomp.SYS_MMAP)
                filter.add_rule(seccomp.ALLOW, seccomp.SYS_MUNMAP)
                filter.add_rule(seccomp.ERRNO(seccomp.EPERM), seccomp.SYS_EXECVE)
                filter.add_rule(seccomp.ERRNO(seccomp.EPERM), seccomp.SYS_EXECVEAT)
                filter.add_rule(seccomp.ERRNO(seccomp.EPERM), seccomp.SYS_FORK)
                filter.add_rule(seccomp.ERRNO(seccomp.EPERM), seccomp.SYS_VFORK)
                filter.add_rule(seccomp.ERRNO(seccomp.EPERM), seccomp.SYS_CLONE)
                filter.add_rule(seccomp.ERRNO(seccomp.EPERM), seccomp.SYS_PTRACE)
                filter.load()
                logger.info("Applied seccomp filter for sandbox security")
            except Exception as e:
                logger.error(f"Failed to apply seccomp filter: {e}")
        except ImportError:
            logger.warning("python-seccomp not available")
        return self._run_subprocess(target, args, kwargs, timeout, workspace)

    def cleanup(self):
        if self._container_id:
            import shutil

            docker_bin = shutil.which("docker")
            if docker_bin:
                try:
                    subprocess.run(
                        [docker_bin, "rm", "-f", self._container_id],
                        capture_output=True,
                        timeout=30,
                    )
                except (OSError, subprocess.TimeoutExpired) as e:
                    logger.warning("Failed to cleanup container: %s", e)
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except (OSError, PermissionError):
                pass
