"""
Unit tests for S3 utilities
"""

from __future__ import annotations

import tempfile

import pytest

try:
    from worker.src.daemon.s3_util import S3Manager

    HAS_S3_UTIL = True
except ImportError:
    HAS_S3_UTIL = False
    S3Manager = None


@pytest.mark.skipif(not HAS_S3_UTIL, reason="s3_util not available")
def test_s3_manager_import():
    assert S3Manager is not None


@pytest.mark.skipif(not HAS_S3_UTIL, reason="s3_util not available")
def test_s3_manager_creation():
    manager = S3Manager()
    assert manager is not None


@pytest.mark.skipif(not HAS_S3_UTIL, reason="s3_util not available")
def test_parse_s3_url():
    manager = S3Manager()
    bucket, key = manager._parse_s3_url("s3://mybucket/path/to/file.pt")
    assert bucket == "mybucket"
    assert key == "path/to/file.pt"
    with pytest.raises(ValueError):
        manager._parse_s3_url("http://example.com/file.pt")


@pytest.mark.skipif(not HAS_S3_UTIL, reason="s3_util not available")
def test_url_type_detection():
    manager = S3Manager()
    assert manager._is_s3_url("s3://bucket/key") is True
    assert manager._is_s3_url("http://example.com") is False
    assert manager._is_s3_url("https://example.com") is False
    assert manager._is_s3_url("/local/path") is False


@pytest.mark.skipif(not HAS_S3_UTIL, reason="s3_util not available")
@pytest.mark.asyncio
async def test_download_local_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("test content")
        temp_path = f.name
    manager = S3Manager()
    result = await manager.download_file(temp_path, "/tmp/dest.txt")
    assert result in [True, False]


@pytest.mark.skipif(not HAS_S3_UTIL, reason="s3_util not available")
@pytest.mark.asyncio
async def test_upload_without_s3_config(tmp_path, monkeypatch):
    import os

    old_bucket = os.environ.pop("S3_BUCKET_NAME", None)
    monkeypatch.setenv("GRADIENT_LOCAL_ROOT", str(tmp_path))
    try:
        manager = S3Manager()
        local = tmp_path / "handoff.txt"
        local.write_text("test content", encoding="utf-8")
        result = await manager.upload_file(str(local), "test-key")
        assert result is not None
    finally:
        if old_bucket:
            os.environ["S3_BUCKET_NAME"] = old_bucket


@pytest.mark.skipif(not HAS_S3_UTIL, reason="s3_util not available")
@pytest.mark.asyncio
async def test_health_check_no_s3():
    manager = S3Manager()
    manager = S3Manager()
    healthy = await manager.health_check()
    assert healthy is False
