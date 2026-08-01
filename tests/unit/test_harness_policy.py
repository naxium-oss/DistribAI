"""Unit tests for fast-test / mock-storage harness detection."""

from __future__ import annotations

import pytest

from services_python.harness_policy import skip_presigned_s3_urls


@pytest.mark.unit
def test_skip_presigned_when_fast_test_and_mock_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISTRIBAI_FAST_TEST", "1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "mock_key")
    assert skip_presigned_s3_urls() is True


@pytest.mark.unit
def test_presigned_allowed_without_fast_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISTRIBAI_FAST_TEST", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "mock_key")
    assert skip_presigned_s3_urls() is False


@pytest.mark.unit
def test_presigned_allowed_with_real_key_in_fast_test(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISTRIBAI_FAST_TEST", "1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
    assert skip_presigned_s3_urls() is False
