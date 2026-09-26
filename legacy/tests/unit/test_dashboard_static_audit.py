"""Static checks for worker dashboard HTML (placeholders and debug patterns)."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DASHBOARD_STATIC = _REPO_ROOT / "worker" / "src" / "dashboard" / "static"

_SUSPICIOUS_PATTERNS = [
    re.compile(r"javascript:void\(0\)", re.IGNORECASE),
    re.compile(r"alert\(", re.IGNORECASE),
    re.compile(r"console\.log\(", re.IGNORECASE),
    re.compile(r"TODO|FIXME|XXX", re.IGNORECASE),
    re.compile(r'placeholder="[^"]*\.\.\.+"', re.IGNORECASE),
    re.compile(r'placeholder="[^"]*enter[^"]*"', re.IGNORECASE),
]


def test_dashboard_static_html_avoids_suspicious_patterns() -> None:
    assert _DASHBOARD_STATIC.is_dir(), f"Missing dashboard static dir: {_DASHBOARD_STATIC}"
    html_files = sorted(_DASHBOARD_STATIC.rglob("*.html"))
    assert html_files, "expected at least one dashboard html file"

    failures: list[str] = []
    for html_path in html_files:
        content = html_path.read_text(encoding="utf-8")
        for pattern in _SUSPICIOUS_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                failures.append(
                    f"{html_path.name}: {pattern.pattern!r} ({len(matches)} occurrence(s))"
                )
    assert not failures, "Issues:\n" + "\n".join(failures)
