"""Tests for gradient/blob URL allowlist policy."""

from __future__ import annotations

from pathlib import Path

from services_python.blob_url_policy import is_allowed_gradient_url


def test_blocks_arbitrary_https_host(monkeypatch):
    monkeypatch.setenv("ALLOWED_BLOB_HOSTS", "127.0.0.1,localhost")
    assert is_allowed_gradient_url("https://evil.example.com/gradient.json") is False


def test_allows_localhost_https(monkeypatch):
    monkeypatch.setenv("ALLOWED_BLOB_HOSTS", "127.0.0.1,localhost")
    assert is_allowed_gradient_url("https://localhost/gradient.json") is True


def test_allows_runtime_file_path(monkeypatch, tmp_path):
    repo_runtime = Path(__file__).resolve().parents[2] / "runtime"
    monkeypatch.setenv("GRADIENT_LOCAL_ROOT", str(repo_runtime))
    payload = repo_runtime / "smoke" / "grad.json"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_text("{}", encoding="utf-8")
    assert is_allowed_gradient_url(str(payload)) is True


def test_blocks_file_outside_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("GRADIENT_LOCAL_ROOT", str(tmp_path / "runtime"))
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    assert is_allowed_gradient_url(str(outside)) is False


def test_s3_requires_configured_bucket(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")
    assert is_allowed_gradient_url("s3://my-bucket/gradients/a.json") is True
    assert is_allowed_gradient_url("s3://other-bucket/gradients/a.json") is False


def test_blocks_s3_key_traversal(monkeypatch):
    monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")
    assert is_allowed_gradient_url("s3://my-bucket/gradients/../secrets/a.json") is False
