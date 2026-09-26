"""Operator-facing notices when a trainer cannot be translated.

These strings are product surface: when translation fails closed, the operator
(and their admin) must know exactly what happened and what to do next:

  * the trainer may use a feature the translator should learn. Ask a
    developer or an AI agent to add support for *that* feature *here*;
  * the trainer script itself may be broken, so it needs a bug fix;
  * pull requests are welcome so everyone gets the improvement; and
  * a GitHub issue will be patched, but it takes a while.

Keep the wording stable: the CLI, the dashboards and the docs all render it.
"""

from __future__ import annotations

from .errors import TranslationReport
from .frameworks import framework_hint

TOOL_PATH = "tools/trainer_translate"
ISSUES_URL = "https://github.com/naxium-oss/DistribAI/issues"

_LINE = "=" * 72


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"    - {i}" for i in items) if items else "    - (unspecified)"


def _framework_lines(report: TranslationReport) -> list[str]:
    """Name a recognized framework and its single-rank mapping, when known."""
    fw = (report.extras or {}).get("framework")
    if not fw:
        return []
    name = fw.get("display") or fw.get("key") or "a training framework"
    lines = ["", f"Detected framework : {name} (single-rank adapter)"]
    if fw.get("detected_via"):
        lines.append(f"Detected via       : {fw['detected_via']}")
    hint = framework_hint(str(fw.get("key") or ""))
    if hint:
        lines.append(f"Single-rank mapping: {hint}")
    return lines


def unsupported_notice(report: TranslationReport) -> str:
    """The trainer is valid Python but uses a feature we cannot translate yet."""
    missing = report.missing_features or ["an unrecognized training construct"]
    return "\n".join(
        [
            _LINE,
            "TRANSLATION FAILED: LibTorch is the only live train path.",
            _LINE,
            f"Trainer : {report.trainer or '<unknown>'}",
            f"Stage   : {report.stage}",
            f"Status  : unsupported ({report.status})",
            "",
            "This trainer uses a feature the translator does not support yet:",
            _bullet_list(missing),
        ]
        + _framework_lines(report)
        + [
            "",
            "What to do:",
            "  1. Your admin may need to ADD SUPPORT for the feature listed above.",
            f"     Support is added in {TOOL_PATH}/ (the translator), or in",
            "     tools/cpp_port/torch/ (the LibTorch runner) if the feature needs",
            "     a new runtime capability.",
            "  2. Ask a developer or an AI agent to add support for:",
            f"       {', '.join(missing)}",
            f"     pointed at {TOOL_PATH} in this repo.",
            "  3. PRs are welcome. If you (or a developer you ask) add the",
            "     support, opening a pull request gets the feature to everyone",
            "     instead of just this job.",
            f"  4. If you cannot act now, open an issue at {ISSUES_URL}.",
            "     It will be patched, but it takes a while.",
            "",
            "Nothing was emitted, so no partial job was written.",
            _LINE,
        ]
    )


def bugfix_notice(report: TranslationReport) -> str:
    """The trainer raised while running: most likely a bug in the script."""
    detail = report.bug_summary or report.detail or "the script raised an error"
    return "\n".join(
        [
            _LINE,
            "TRANSLATION FAILED: the trainer looks broken, not just unsupported.",
            _LINE,
            f"Trainer : {report.trainer or '<unknown>'}",
            f"Stage   : {report.stage}",
        ]
        + _framework_lines(report)
        + [
            "",
            "The script raised an error while the translator was inspecting it:",
            f"    {detail}",
            "",
            "What to do:",
            "  1. This script needs a BUG FIX before it can be translated. Fix the",
            "     error, then resubmit. The translator retries from scratch.",
            "  2. If you believe the script is correct and the translator is wrong,",
            f"     ask a developer or an AI agent to add support for it in {TOOL_PATH}/,",
            "     including the traceback below.",
            "  3. PRs are suggested so others hit by the same bug get the fix too.",
            f"  4. Otherwise open an issue at {ISSUES_URL}. It will be patched,",
            "     but it takes a while.",
        ]
        + (["", "Traceback (tail):", _indent_tail(report.traceback)] if report.traceback else [])
        + ["", "Nothing was emitted, so no partial job was written.", _LINE]
    )


def _indent_tail(tb: str, max_lines: int = 12) -> str:
    lines = tb.strip().splitlines()[-max_lines:]
    return "\n".join(f"    {ln}" for ln in lines)


def render_report(report: TranslationReport) -> str:
    """Choose the right notice for a report."""
    if report.status == "needs_bugfix":
        return bugfix_notice(report)
    return unsupported_notice(report)
