"""The Python train path is PARKED. LibTorch is the only live train path.

The native LibTorch runner (``tools/cpp_port/torch/torch_train.cpp``) is what the
grid executes for jobs. This module fails closed for any attempt to dispatch a
training job through the frozen Python stack, with an operator-facing notice.

The only way to run the legacy path is an explicit, deliberate opt-in:

    DISTRIBAI_ALLOW_LEGACY_TRAIN=1

...which exists for debugging the frozen stack itself, never for real jobs.

To run a Python trainer today, translate it first::

    python -m trainer_translate.translate trainer.py --out jobs/my-job
    build/cpp_port/distribai_torch_train --spec jobs/my-job/job.json --json
"""

from __future__ import annotations

import os

ALLOW_ENV = "DISTRIBAI_ALLOW_LEGACY_TRAIN"
_TRUTHY = {"1", "true", "yes", "on"}

TOOL_PATH = "tools/trainer_translate"
ISSUES_URL = "https://github.com/naxium-oss/DistribAI/issues"

_LINE = "=" * 72


class LegacyTrainPathDisabledError(RuntimeError):
    """Raised when a job is dispatched through the parked Python train path."""

    def __init__(self, where: str):
        self.where = where
        self.notice = legacy_train_disabled_notice(where)
        super().__init__(self.notice)


def legacy_train_enabled() -> bool:
    """True only when an operator explicitly opted the legacy path back in."""
    return os.environ.get(ALLOW_ENV, "").strip().lower() in _TRUTHY


def legacy_train_disabled_notice(where: str) -> str:
    return "\n".join(
        [
            _LINE,
            "LEGACY PYTHON TRAIN PATH IS PARKED. LibTorch is the only live train path.",
            _LINE,
            f"Dispatch point : {where}",
            "The Python distributed stack under legacy/ is frozen and is NOT used for jobs.",
            "",
            "What to do:",
            "  1. Submit a Python trainer through the translator instead:",
            f"       python -m {TOOL_PATH}.translate <trainer.py> --out <jobdir>",
            "     then run the emitted job with build/cpp_port/distribai_torch_train.",
            "  2. If your trainer needs a feature the translator does not support yet,",
            "     your admin can add support for it (ask a developer or an AI agent to add",
            f"     that feature in {TOOL_PATH}/ or tools/cpp_port/torch/). PRs are suggested",
            "     so others get the feature too.",
            f"  3. Otherwise open an issue at {ISSUES_URL}, it will be patched, but",
            "     it takes a while.",
            f"  4. Debugging the frozen stack itself? Set {ALLOW_ENV}=1 to bypass this guard.",
            _LINE,
        ]
    )


def require_modern_train_path(where: str) -> None:
    """Raise unless the caller is doing an explicitly-opted-in debug run.

    Called at every point a training job could be handed to the legacy stack.
    """
    if legacy_train_enabled():
        return
    raise LegacyTrainPathDisabledError(where)
