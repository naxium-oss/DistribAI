"""CORS origin policy for the orchestrator admin HTTP API."""

from __future__ import annotations

import os

_LOCAL_DASHBOARD_ORIGINS = (
    "http://127.0.0.1:3210",
    "http://localhost:3210",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
)


def cors_origins_list() -> list[str]:
    """Effective allowlist: local dashboard ports if unset; never silently becomes *."""
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if raw == "*":
        return ["*"]
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return list(_LOCAL_DASHBOARD_ORIGINS)


def cors_is_permissive() -> bool:
    """True when CORS_ALLOWED_ORIGINS explicitly includes *."""
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return False
    return raw == "*" or any(part.strip() == "*" for part in raw.split(","))
