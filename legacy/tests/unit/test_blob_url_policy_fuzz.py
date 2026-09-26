"""Fuzz-style parametrized cases for gradient/blob URL allowlist."""

from __future__ import annotations

import pytest

from services_python.blob_url_policy import is_allowed_gradient_url, sanitize_s3_object_key


@pytest.fixture(autouse=True)
def _localhost_hosts(monkeypatch):
    monkeypatch.setenv("ALLOWED_BLOB_HOSTS", "127.0.0.1,localhost")
    monkeypatch.setenv("S3_BUCKET_NAME", "my-bucket")


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "javascript:alert(1)",
        "data:text/plain,evil",
        "gopher://127.0.0.1/1",
        "https://evil.example/gradient.json",
        "https://127.0.0.1.evil.com/gradient.json",
        "https://localhost.evil.com/gradient.json",
        "s3://other-bucket/gradients/a.json",
        "s3://my-bucket/",
        "s3:///no-bucket/path",
        "http://[::1]/gradient.json",
        "https://user:pass@127.0.0.1/gradient.json",
    ],
)
def test_fuzz_disallowed_urls(url: str):
    assert is_allowed_gradient_url(url) is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/gradient.json",
        "https://localhost/gradient.json",
        "http://localhost:8766/weights/x.pt",
        "s3://my-bucket/gradients/a.json",
    ],
)
def test_fuzz_allowed_urls(url: str):
    assert is_allowed_gradient_url(url) is True


@pytest.mark.unit
def test_fuzz_rejects_path_traversal_outside_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("GRADIENT_LOCAL_ROOT", str(tmp_path / "runtime"))
    outside = tmp_path / "escape.json"
    outside.write_text("{}", encoding="utf-8")
    traversal = str(outside) + "/../escape.json"
    assert is_allowed_gradient_url(traversal) is False


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    [
        "../escape.bin",
        "gradients/../../secret.bin",
        "",
        "/leading.bin",
    ],
)
def test_sanitize_s3_object_key_rejects_traversal(raw: str):
    assert sanitize_s3_object_key(raw) is None


@pytest.mark.unit
def test_sanitize_s3_object_key_normalizes_slashes():
    assert sanitize_s3_object_key("gradients\\ok\\file.bin") == "gradients/ok/file.bin"


def test_fuzz_upload_key_traversal_not_used_for_local_allowlist(monkeypatch, tmp_path):
    """S3 keys with .. are blocked at upload construction; local paths must resolve in-root."""
    monkeypatch.setenv("GRADIENT_LOCAL_ROOT", str(tmp_path / "runtime"))
    nested = tmp_path / "runtime" / "ok.json"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("{}", encoding="utf-8")
    assert is_allowed_gradient_url(str(nested)) is True
    assert is_allowed_gradient_url(str(tmp_path / "runtime" / ".." / ".." / "escape.json")) is False
