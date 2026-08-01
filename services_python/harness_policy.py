"""Detect pytest / local harness environments that must not call real S3."""

from __future__ import annotations

import os

from services_python.env_bool import env_truthy

_MOCK_AWS_KEYS = frozenset(
    {
        "mock_key",
        "mock",
        "fake",
        "testing",
        "test",
    }
)


def skip_presigned_s3_urls() -> bool:
    """Return True when presigned S3 URLs would point at unreachable mock storage."""
    if env_truthy("DISTRIBAI_FAST_TEST") is not True:
        return False
    key = os.getenv("AWS_ACCESS_KEY_ID", "").strip().lower()
    if not key:
        return True
    if key in _MOCK_AWS_KEYS or key.startswith("mock_"):
        return True
    return False


def harness_disables_s3() -> bool:
    """True when pytest/fast harness must not call real S3 (avoids .env bleed)."""
    return env_truthy("DISTRIBAI_FAST_TEST") is True
