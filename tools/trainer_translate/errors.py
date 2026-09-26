"""Fail-closed error types for the Python -> C++/LibTorch translator.

Every failure carries a :class:`TranslationReport` so the CLI can both (a) exit
non-zero, and (b) print an actionable notice for the node operator / admin.
Nothing is emitted to the job directory unless translation fully succeeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranslationReport:
    """Structured reason a trainer could not be translated.

    status:
      "unsupported": the trainer is valid Python but uses a feature the
                      translator (or the LibTorch live path) does not support.
                      The admin may need to add support for it.
      "needs_bugfix": the trainer itself raised while running, so it looks like a
                      script bug rather than a missing feature.
      "internal": the translator hit an unexpected error. Report it upstream.
    """

    status: str
    stage: str  # import | capture | script | analyze | emit
    missing_features: list[str] = field(default_factory=list)
    bug_summary: str | None = None
    traceback: str | None = None
    detail: str | None = None
    trainer: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage": self.stage,
            "missing_features": list(self.missing_features),
            "bug_summary": self.bug_summary,
            "traceback": self.traceback,
            "detail": self.detail,
            "trainer": self.trainer,
            "extras": self.extras,
        }


class TranslationError(RuntimeError):
    """Raised when a trainer cannot be translated; carries a report."""

    def __init__(self, report: TranslationReport):
        self.report = report
        super().__init__(report.detail or report.status)

    @property
    def status(self) -> str:
        return self.report.status
