"""Sandbox backend factory and public exports.

Exports:
* :class:`SandboxBackend` / :class:`SandboxResult` / :class:`NetworkPolicy`
* :func:`detect_backend` / :func:`build_sandbox`

Pick order: env override → docker (PATH) → nsjail (POSIX) → subprocess.
``build_sandbox`` never raises; it warns and falls back so the daemon
keeps serving jobs.
"""

from __future__ import annotations

import logging
import os
import shutil

from .base import NetworkPolicy, SandboxBackend, SandboxResult
from .docker import DockerSandbox
from .nsjail import NsjailSandbox
from .subprocess_backend import SubprocessSandbox

__all__ = [
    "DockerSandbox",
    "NetworkPolicy",
    "NsjailSandbox",
    "SandboxBackend",
    "SandboxResult",
    "SubprocessSandbox",
    "build_sandbox",
    "detect_backend",
]

logger = logging.getLogger(__name__)

_VALID_NAMES = {"docker", "nsjail", "subprocess"}


def detect_backend() -> str:
    """Name of the preferred backend for this machine.

    Env override wins. Otherwise cheap PATH probes; live availability is
    re-checked in :meth:`SandboxBackend.is_available` / ``build_sandbox``.
    """
    override = os.getenv("DISTRIBAI_SANDBOX_BACKEND")
    if override:
        override = override.strip().lower()
        if override in _VALID_NAMES:
            return override
        logger.warning(
            "Ignoring invalid DISTRIBAI_SANDBOX_BACKEND=%r (must be one of %s).",
            override,
            sorted(_VALID_NAMES),
        )

    if shutil.which("docker"):
        return "docker"
    if shutil.which("nsjail") and os.name == "posix":
        return "nsjail"
    return "subprocess"


def build_sandbox(
    backend: str | None = None,
    **kwargs,
) -> SandboxBackend:
    """Construct a usable backend, degrading when probes fail.

    Extra ``kwargs`` go to the backend constructor (e.g. Docker ``image=``).
    """
    name = (backend or detect_backend()).lower()

    if name == "docker":
        candidate = DockerSandbox(**kwargs)
        if candidate.is_available():
            return candidate
        logger.warning(
            "DockerSandbox unavailable (daemon not reachable?); "
            "falling back to next backend."
        )
        # Prefer nsjail next, else subprocess
        if shutil.which("nsjail") and os.name == "posix":
            return NsjailSandbox()
        return SubprocessSandbox()

    if name == "nsjail":
        candidate = NsjailSandbox(**kwargs)
        if candidate.is_available():
            return candidate
        logger.warning(
            "NsjailSandbox unavailable on this host; "
            "falling back to SubprocessSandbox."
        )
        return SubprocessSandbox()

    return SubprocessSandbox(**kwargs)
