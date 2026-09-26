"""Shared environment boolean parsing."""

from __future__ import annotations

import os


def env_truthy(name: str) -> bool | None:
    raw = os.getenv(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return None
