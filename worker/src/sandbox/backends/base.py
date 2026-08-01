"""Abstract ``SandboxBackend`` contract for ScriptRunner isolation.

Concrete implementations:

* ``DockerSandbox`` — short-lived container, RO root + tmpfs, dropped
  caps, netns, hard memory/CPU limits.
* ``NsjailSandbox`` — Linux ``nsjail`` with mount/user/net namespaces
  and seccomp.
* ``SubprocessSandbox`` — portable v1.1 hardened child (POSIX rlimits,
  env allow-list, ``PR_SET_NO_NEW_PRIVS``) used when Docker/nsjail are
  absent.

``build_sandbox`` picks a backend from ``DISTRIBAI_SANDBOX_BACKEND`` or
auto-detection.

``run_script`` takes a prepared ``task_dir`` (``run.py``, hyperparams,
optional site-packages) and a filtered ``env``, returning
``SandboxResult`` so ``script_runner`` stays backend-agnostic.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NetworkPolicy(Enum):
    """Egress policy requested for a single ``run_script`` call.

    * ``NONE`` — no outbound net. Docker: ``--network=none``; nsjail:
      ``clone_newnet``; subprocess cannot enforce (use a real backend).
    * ``RESTRICTED`` — intended HF/S3 allow-list; today approximated as
      DNS + TCP 443 until host iptables land in v1.3.
    * ``OPEN`` — unrestricted egress (legacy v1.1); trusted jobs only.
    """

    NONE = "none"
    RESTRICTED = "restricted"
    OPEN = "open"


@dataclass
class SandboxResult:
    """Result record produced by every backend ``run_script`` call.

    Backends trim stdout/stderr to roughly 10 KB each to match the
    ScriptRunner contract; full logs belong under ``task_dir/output``.
    """

    return_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool
    backend_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SandboxBackend(ABC):
    """Base class every isolation backend subclasses.

    Required: :meth:`run_script`. Optional: override :meth:`is_available`
    for live probes (e.g. ``docker info``) beyond PATH checks.
    """

    name: str = "abstract"

    def is_available(self) -> bool:
        """Whether this host can actually execute jobs with this backend.

        Default is True; Docker/nsjail overrides probe their daemons/binaries.
        """
        return True

    @abstractmethod
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
        """Run ``task_dir/run.py`` under this backend's isolation.

        Parameters
        ----------
        task_dir
            Prepared workspace with ``run.py``, hyperparams, and optional
            ``.site-packages/``. Must be visible inside the sandbox.
        env
            Pre-filtered environment; do not pull extra secrets from
            ``os.environ``.
        max_runtime_seconds
            Wall-clock kill (Docker kill / nsjail time_limit /
            subprocess communicate timeout).
        max_memory_mb
            Hard RSS/AS cap (``--memory``, ``rlimit_as``, ``RLIMIT_AS``).
        max_cpu_time_sec
            CPU-time cap distinct from wall clock.
        network
            :class:`NetworkPolicy` request.

        Returns
        -------
        SandboxResult
            Exit code (``-1`` if killed early); ``timed_out`` when the
            wall-clock limit fired.
        """
        raise NotImplementedError
