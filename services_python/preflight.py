"""Submission-time checks for script job packages (CLI and admin HTTP)."""

from __future__ import annotations

import io
import re
import tarfile
from typing import Any

FORBIDDEN_ARCHIVE_PATTERNS = (
    re.compile(r"(^|/)\.env($|/)"),
    re.compile(r"(^|/)id_rsa"),
    re.compile(r"(^|/)\.ssh/"),
    re.compile(r"\.pem$"),
    re.compile(r"\.key$"),
)
MAX_SCRIPT_BYTES = 5_000_000


def validate_script_tarball(data: bytes) -> tuple[bool, str | None, dict[str, Any]]:
    """Inspect a gzipped tar script bundle; return (ok, error_message, metadata)."""
    if not data:
        return False, "empty script package", {}
    if len(data) > MAX_SCRIPT_BYTES:
        return False, "script package too large", {"size_bytes": len(data)}

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            member_names = [member.name for member in archive.getmembers() if member.isfile()]
    except tarfile.TarError as exc:
        return False, f"invalid tarball: {exc}", {}

    if not member_names:
        return False, "tarball contains no files", {}

    def _is_entry(name: str, basename: str) -> bool:
        return name == basename or name.endswith(f"/{basename}")

    has_run_py = any(_is_entry(name, "run.py") for name in member_names)
    has_run_ipynb = any(
        _is_entry(name, "run.ipynb") or name.endswith(".ipynb") for name in member_names
    )
    if not has_run_py and not has_run_ipynb:
        return (
            False,
            "entry script run.py (or run.ipynb) missing from package",
            {"members": len(member_names)},
        )

    for member_name in member_names:
        for blocked in FORBIDDEN_ARCHIVE_PATTERNS:
            if blocked.search(member_name):
                return False, f"forbidden path in bundle: {member_name}", {"path": member_name}

    return True, None, {"member_count": len(member_names), "size_bytes": len(data)}
