"""Integration test defaults (fast poll/timeouts when DISTRIBAI_FAST_TEST=1)."""

from __future__ import annotations

import os

import pytest

from tests.fast_env import fast_mode_enabled, poll_seconds, wait_seconds


@pytest.fixture(scope="session", autouse=True)
def _integration_storage_env():
    """Align mock S3 + blob allowlist with integration harness expectations."""
    os.environ.setdefault("S3_BUCKET_NAME", "test-bucket")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "mock_key")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "mock_secret")
    os.environ.setdefault(
        "ALLOWED_BLOB_HOSTS",
        "127.0.0.1,localhost,test-bucket.s3.amazonaws.com,s3.amazonaws.com",
    )
    yield


@pytest.fixture(scope="module", autouse=True)
def _fast_integration_orch_warmup():
    """Module orch fixtures use time.sleep(1); cap via session time.sleep patch in root conftest."""
    yield


def integration_timeout(default: float) -> float:
    if not fast_mode_enabled():
        return default
    if default >= 60.0:
        return 15.0
    if default >= 30.0:
        return 12.0
    if default >= 15.0:
        return 8.0
    return min(default, 4.0)


def integration_poll(default: float = 0.5) -> float:
    return poll_seconds(default)


def orch_startup_delay() -> float:
    return wait_seconds(1.0)
