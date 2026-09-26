"""Deprecation notice for the Python control plane.

The gRPC orchestrator and the Python worker daemon are deprecated: the C++
coordinator in ``tools/cpp_port/grid`` replaces both, runs jobs through LibTorch,
and needs no interpreter on the train path. This module exists so an operator who
starts the old stack gets told once, in plain words, instead of discovering it
from a README months later.

Nothing here blocks anything. It is a notice, not a gate. The hard gate on
dispatching training work lives in ``services_python.train_path_guard``.
"""

from __future__ import annotations

import os
import sys

# Suppress the notice for scripted runs that already know. Anything other than
# the two values below is treated as "warn anyway".
SILENCE_ENV = "DISTRIBAI_SILENCE_DEPRECATION"

NOTICE = """\
DistribAI: the Python control plane is deprecated.
  Replace this with the C++ coordinator and worker:
      build/cpp_port/distribai_orch      # registry, queue, aggregation, dashboard
      build/cpp_port/distribai_worker    # runs a replica under rlimits
  Docs: docs/from-python-to-cpp.md
  This stack keeps working for now. It receives bug fixes only, no new features.
  Set {env}=1 to silence this notice.
"""


def _silenced() -> bool:
    return os.environ.get(SILENCE_ENV, "").strip() in {"1", "true", "yes", "on"}


def notice(component: str) -> str:
    """The notice text, tagged with the component that started."""
    return f"[{component}] " + NOTICE.format(env=SILENCE_ENV)


def warn(component: str) -> None:
    """Write the notice to stderr unless it is silenced."""
    if _silenced():
        return
    print(notice(component), file=sys.stderr, flush=True)


def warn_legacy_orchestrator() -> None:
    warn("orchestrator_grpc")


def warn_legacy_worker() -> None:
    warn("worker.daemon")


__all__ = [
    "NOTICE",
    "SILENCE_ENV",
    "notice",
    "warn",
    "warn_legacy_orchestrator",
    "warn_legacy_worker",
]
