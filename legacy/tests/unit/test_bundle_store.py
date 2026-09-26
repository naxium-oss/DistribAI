"""Unit tests for on-disk script bundle storage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services_python.bundle_store import delete_bundle, load_bundle, save_bundle


@pytest.mark.unit
def test_bundle_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_BUNDLE_DIR", str(tmp_path))
    payload = b"\x1f\x8b\x08" + b"fake-tar-gz-bytes"
    save_bundle("task-save-1", payload)
    assert load_bundle("task-save-1") == payload
    assert delete_bundle("task-save-1") is True
    assert load_bundle("task-save-1") is None


@pytest.mark.unit
def test_bundle_store_rejects_bad_task_id(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_BUNDLE_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        save_bundle("../escape", b"x")
    assert load_bundle("../escape") is None


@pytest.mark.unit
def test_bundle_store_rejects_empty_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_BUNDLE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        save_bundle("task-empty", b"")


@pytest.mark.unit
def test_load_bundle_falls_back_to_s3(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_BUNDLE_DIR", str(tmp_path))
    payload = b"\x1f\x8b\x08-s3"
    mock_client = MagicMock()
    mock_client.get_object.return_value = {"Body": MagicMock(read=lambda: payload)}
    mock_client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
    with patch(
        "services_python.bundle_store._s3_client_and_bucket",
        return_value=(mock_client, "test-bucket"),
    ):
        assert load_bundle("task-s3-1") == payload


@pytest.mark.unit
def test_save_bundle_uploads_to_s3(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_BUNDLE_DIR", str(tmp_path))
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
    payload = b"\x1f\x8b\x08-upload"
    mock_client = MagicMock()
    with patch(
        "services_python.bundle_store._s3_client_and_bucket",
        return_value=(mock_client, "test-bucket"),
    ):
        save_bundle("task-up-1", payload)
    mock_client.put_object.assert_called_once()
    call = mock_client.put_object.call_args.kwargs
    assert call["Bucket"] == "test-bucket"
    assert call["Key"] == "bundles/task-up-1.tar.gz"
    assert call["Body"] == payload
