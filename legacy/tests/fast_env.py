"""Shared timing knobs for pytest (fast local/CI gate)."""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def fast_mode_enabled() -> bool:
    return os.getenv("DISTRIBAI_FAST_TEST", "1").strip().lower() in _TRUTHY


def poll_seconds(default: float) -> float:
    if not fast_mode_enabled():
        return default
    override = os.getenv("DISTRIBAI_TEST_POLL", "").strip()
    if override:
        return float(override)
    return min(default, 0.05)


def startup_seconds(default: float) -> float:
    if not fast_mode_enabled():
        return default
    override = os.getenv("DISTRIBAI_TEST_STARTUP", "").strip()
    if override:
        return float(override)
    return min(default, 0.25)


def wait_seconds(default: float) -> float:
    """Cap long fixed sleeps in integration/e2e when fast mode is on."""
    if not fast_mode_enabled():
        return default
    override = os.getenv("DISTRIBAI_TEST_WAIT", "").strip()
    if override:
        return float(override)
    if default >= 2.0:
        return 0.2
    if default >= 0.5:
        return 0.08
    return default
