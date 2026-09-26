"""trainer_translate: turn a Python trainer into a LibTorch job.

The native C++ port is the only live train path. This package turns a
user-supplied Python training script (any architecture, custom or weird) into
a C++/LibTorch distributed job the grid can run:

    jobdir/
      model.pt   TorchScript module (the captured architecture)
      job.json   optimizer / loss / steps / shapes / metadata
      notes.md   human-readable translation notes (warnings, provenance)

Translation is **fail-closed**: if any part of the trainer cannot be translated
we emit no job and exit non-zero with a structured report telling the operator
exactly what is missing and how to get support. See ``notices.py``.
"""

from .errors import TranslationError, TranslationReport
from .frameworks import FRAMEWORKS, FrameworkSpec, detect_framework, install_single_rank
from .notices import (
    ISSUES_URL,
    TOOL_PATH,
    bugfix_notice,
    render_report,
    unsupported_notice,
)
from .translate import translate as translate_trainer

__all__ = [
    "FRAMEWORKS",
    "ISSUES_URL",
    "TOOL_PATH",
    "FrameworkSpec",
    "TranslationError",
    "TranslationReport",
    "bugfix_notice",
    "detect_framework",
    "install_single_rank",
    "unsupported_notice",
    "render_report",
    "translate_trainer",
]
