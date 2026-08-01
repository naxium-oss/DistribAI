"""Lightweight static validation for submitted grid training scripts."""

from __future__ import annotations

import ast
from typing import Any

_DISALLOWED_NAMES = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "input",
        "breakpoint",
    }
)


def validate_submitted_script(source: str | None) -> tuple[list[str], list[str]]:
    """Parse script and return (error_codes, human_suggestions).

    This does not guarantee runtime safety; it catches obvious issues early.
    """
    errors: list[str] = []
    suggestions: list[str] = []

    if source is None:
        return errors, suggestions

    stripped = source.strip()
    if not stripped:
        errors.append("empty_script")
        suggestions.append("Provide non-empty Python source in script_content.")
        return errors, suggestions

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        errors.append(f"syntax_error:{exc.lineno}:{exc.msg}")
        suggestions.append("Fix the Python syntax error at the indicated line.")
        return errors, suggestions

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DISALLOWED_NAMES:
                errors.append(f"disallowed_call:{func.id}")
                suggestions.append(
                    f"Avoid {func.id}() in grid scripts; use safe APIs and approved I/O patterns."
                )
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".", 1)[0]
                if mod in ("subprocess", "ctypes", "multiprocessing"):
                    errors.append(f"disallowed_import:{alias.name}")
                    suggestions.append(
                        "Subprocess and native FFI are restricted on worker nodes; "
                        "use PyTorch/torch.distributed and standard libraries where possible."
                    )
        if isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".", 1)[0]
                if mod in ("subprocess", "ctypes", "multiprocessing"):
                    errors.append(f"disallowed_import_from:{node.module}")
                    suggestions.append(
                        "Subprocess and native FFI imports are restricted on worker nodes."
                    )

    return errors, suggestions


def validation_summary(errors: list[str], suggestions: list[str]) -> dict[str, Any]:
    return {"validation_errors": errors, "suggestions": list(dict.fromkeys(suggestions))}
