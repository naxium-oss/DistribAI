"""Tests for services_python.auto_update.UpdateService."""

from __future__ import annotations

import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from services_python.auto_update import UpdateService


@pytest.fixture
def update_service() -> UpdateService:
    return UpdateService(
        update_url="https://github.com/test-org/example/releases/latest/download",
        current_version="1.0.0",
    )


def test_update_service_initialization(update_service: UpdateService) -> None:
    assert update_service.current_version == "1.0.0"
    assert update_service.update_url.endswith("download")
    assert update_service.version_url.endswith("version.json")


@pytest.mark.parametrize(
    ("latest", "current", "expected"),
    [
        ("1.0.1", "1.0.0", True),
        ("1.0.0", "1.0.0", False),
        ("0.9.9", "1.0.0", False),
        ("2.0.0", "1.9.9", True),
    ],
)
def test_is_newer_version(
    update_service: UpdateService, latest: str, current: str, expected: bool
) -> None:
    assert update_service._is_newer_version(latest, current) is expected


def test_check_for_updates_reports_newer_release(
    update_service: UpdateService, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.dumps(
        {
            "version": "1.0.1",
            "download_url": "https://example.com/pkg.zip",
            "size_mb": 1,
            "notes": "test",
            "hash": "abc",
        }
    ).encode()

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return payload

    monkeypatch.setattr(
        "services_python.auto_update.urllib.request.urlopen",
        lambda url, timeout=10: _FakeResponse(),
    )

    info = update_service.check_for_updates()
    assert info["update_available"] is True
    assert info["version"] == "1.0.1"
    assert info["download_url"] == "https://example.com/pkg.zip"


def test_check_for_updates_handles_fetch_errors(
    update_service: UpdateService, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr("services_python.auto_update.urllib.request.urlopen", _raise)
    info = update_service.check_for_updates()
    assert info["update_available"] is False
    assert "error" in info
    assert info["current_version"] == "1.0.0"


def test_check_for_updates_rejects_insecure_download_url(
    update_service: UpdateService, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.dumps(
        {
            "version": "1.0.1",
            "download_url": "http://example.com/pkg.zip",
        }
    ).encode()

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return payload

    monkeypatch.setattr(
        "services_python.auto_update.urllib.request.urlopen",
        lambda url, timeout=10: _FakeResponse(),
    )

    info = update_service.check_for_updates()
    assert info["update_available"] is False
    assert "HTTPS" in info["error"]


def test_download_update_rejects_non_https_url(update_service: UpdateService) -> None:
    ok, message = update_service.download_update("file:///tmp/distribai-update.zip")
    assert ok is False
    assert "HTTPS" in message


def test_verify_update_package_rejects_non_https_signature_url(
    update_service: UpdateService, tmp_path: Path
) -> None:
    package = tmp_path / "package.zip"
    package.write_bytes(b"x" * 2048)

    assert update_service.verify_update_package(str(package), "file:///tmp/package.sig") is False


def test_install_update_rejects_zip_path_traversal(
    update_service: UpdateService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / "install"
    install_root.mkdir()
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "owned")

    monkeypatch.setenv("DISTRIBAI_INSTALL_ROOT", str(install_root))
    ok, message = update_service.install_update(str(archive))

    assert ok is False
    assert "Unsafe archive member path" in message
    assert not (tmp_path / "escape.txt").exists()


def test_install_update_rejects_tar_path_traversal(
    update_service: UpdateService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / "install"
    install_root.mkdir()
    archive = tmp_path / "evil.tar.gz"
    payload = b"owned"

    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(payload)
        tar.addfile(info, BytesIO(payload))

    monkeypatch.setenv("DISTRIBAI_INSTALL_ROOT", str(install_root))
    ok, message = update_service.install_update(str(archive))

    assert ok is False
    assert "Unsafe archive member path" in message
    assert not (tmp_path / "escape.txt").exists()


def test_install_update_accepts_safe_zip(
    update_service: UpdateService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_root = tmp_path / "install"
    install_root.mkdir()
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("app/config.txt", "ok")

    monkeypatch.setenv("DISTRIBAI_INSTALL_ROOT", str(install_root))
    ok, message = update_service.install_update(str(archive))

    assert ok is True, message
    assert (install_root / "app" / "config.txt").read_text(encoding="utf-8") == "ok"


def test_orchestrator_imports_auto_update_module() -> None:
    """Ensure the orchestrator package still loads (no circular import regressions)."""
    import services_python.orchestrator_grpc as orchestrator_grpc  # noqa: PLC0415

    assert orchestrator_grpc is not None
