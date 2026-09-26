"""Worker S3/local blob path allowlist parity with orchestrator policy."""

from __future__ import annotations

import pytest

from worker.src.daemon.s3_util import S3Manager


@pytest.mark.asyncio
async def test_worker_download_blocks_disallowed_https(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLOWED_BLOB_HOSTS", "127.0.0.1,localhost")
    manager = S3Manager()
    dest = tmp_path / "out.bin"
    ok = await manager.download_file("https://evil.example/blob.bin", str(dest))
    assert ok is False
    assert not dest.exists()


@pytest.mark.asyncio
async def test_worker_upload_blocks_disallowed_local_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)
    monkeypatch.setenv("GRADIENT_LOCAL_ROOT", str(tmp_path / "allowed_root"))
    manager = S3Manager()
    outside = tmp_path / "outside_allowed_root.bin"
    outside.write_bytes(b"x")
    url = await manager.upload_file(str(outside), "gradients/x.bin")
    assert url is None


@pytest.mark.asyncio
async def test_worker_upload_blocks_traversal_s3_key(monkeypatch, tmp_path):
    monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    inside = tmp_path / "allowed_root"
    inside.mkdir(parents=True)
    blob = inside / "w.pt"
    blob.write_bytes(b"x")
    manager = S3Manager()
    url = await manager.upload_file(str(blob), "../outside/w.pt")
    assert url is None
