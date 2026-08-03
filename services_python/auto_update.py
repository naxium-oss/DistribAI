#!/usr/bin/env python3
"""
Auto-Update Service for DistribAI Node Applications

Handles version checking, download verification, and update installation
for node applications based on GitHub releases.
"""

import base64
import binascii
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urlparse

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key

logger = logging.getLogger(__name__)

_ALLOWED_UPDATE_SUFFIXES = (".whl", ".zip", ".tar.gz", ".tgz")

# Env var holding the Ed25519 public key that signs release packages. Accepts
# a PEM block or raw 32-byte key (hex or base64). When set, update packages
# MUST carry a valid signature; when unset, verification falls back to a
# best-effort integrity check and logs that signatures are unenforced.
UPDATE_PUBLIC_KEY_ENV = "DISTRIBAI_UPDATE_PUBLIC_KEY"


def _load_update_public_key() -> Ed25519PublicKey | None:
    """Load the release-signing Ed25519 public key from the environment."""
    raw = os.getenv(UPDATE_PUBLIC_KEY_ENV, "").strip()
    if not raw:
        return None
    try:
        if "BEGIN PUBLIC KEY" in raw:
            key = load_pem_public_key(raw.encode())
            if isinstance(key, Ed25519PublicKey):
                return key
            logger.error("%s is not an Ed25519 public key", UPDATE_PUBLIC_KEY_ENV)
            return None
        for decoder in (bytes.fromhex, lambda s: base64.b64decode(s, validate=True)):
            try:
                key_bytes = decoder(raw)
            except (ValueError, binascii.Error):
                continue
            if len(key_bytes) == 32:
                return Ed25519PublicKey.from_public_bytes(key_bytes)
        logger.error("%s must be PEM, or 32-byte hex/base64", UPDATE_PUBLIC_KEY_ENV)
    except (ValueError, TypeError) as exc:
        logger.error("Failed to parse %s: %s", UPDATE_PUBLIC_KEY_ENV, exc)
    return None


def _decode_signature(signature_text: str) -> bytes | None:
    """Decode a detached Ed25519 signature (hex or base64, 64 bytes)."""
    candidate = signature_text.strip()
    for decoder in (bytes.fromhex, lambda s: base64.b64decode(s, validate=True)):
        try:
            decoded = decoder(candidate)
        except (ValueError, binascii.Error):
            continue
        if len(decoded) == 64:
            return decoded
    return None


def _require_https_url(url: str, label: str) -> str:
    """Return a normalized HTTPS URL or raise ValueError."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError(f"{label} must be an HTTPS URL")
    return parsed.geturl()


def _safe_download_filename(download_url: str) -> str:
    parsed = urlparse(download_url)
    filename = PurePosixPath(unquote(parsed.path)).name
    if (
        not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or PureWindowsPath(filename).drive
    ):
        raise ValueError("Download URL must include a safe filename")
    lower = filename.lower()
    if not any(lower.endswith(suffix) for suffix in _ALLOWED_UPDATE_SUFFIXES):
        raise ValueError("Unsupported update package format")
    return filename


def _safe_archive_target(root: Path, member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    win_path = PureWindowsPath(member_name)
    if (
        not path.parts
        or path.is_absolute()
        or win_path.is_absolute()
        or win_path.drive
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"Unsafe archive member path: {member_name}")
    target = (root / Path(*path.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Archive member escapes extraction root: {member_name}") from exc
    return target


def _safe_unpack_archive(path: Path, extract_root: Path) -> None:
    name_lower = path.name.lower()
    if name_lower.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                target = _safe_archive_target(extract_root, info.filename)
                file_mode = (info.external_attr >> 16) & 0o170000
                if file_mode == 0o120000:
                    raise ValueError(f"Refusing archive symlink: {info.filename}")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, open(target, "wb") as dest:
                    shutil.copyfileobj(source, dest)
        return

    if name_lower.endswith(".tar.gz") or name_lower.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                target = _safe_archive_target(extract_root, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Refusing special archive member: {member.name}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Archive member could not be read: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, open(target, "wb") as dest:
                    shutil.copyfileobj(source, dest)
        return

    raise ValueError("Unsupported package format (use .whl, .zip, or .tar.gz)")


class UpdateService:
    """Service for managing automatic updates of DistribAI node applications."""

    def __init__(self, update_url: str, current_version: str = "1.0.0"):
        """
        Initialize the update service.

        Args:
            update_url: Base URL for GitHub releases
            current_version: Current application version
        """
        self.update_url = _require_https_url(update_url, "update_url").rstrip("/")
        self.current_version = current_version
        self.version_url = f"{self.update_url}/version.json"

    def check_for_updates(self) -> dict:
        """
        Check if updates are available.

        Returns:
            Dictionary with update information
        """
        try:
            with urllib.request.urlopen(self.version_url, timeout=10) as response:
                version_info = json.loads(response.read().decode("utf-8"))

            if self._is_newer_version(version_info.get("version", "1.0.0"), self.current_version):
                download_url = version_info.get("download_url")
                if download_url:
                    download_url = _require_https_url(download_url, "download_url")
                return {
                    "update_available": True,
                    "version": version_info.get("version"),
                    "download_url": download_url,
                    "size_mb": version_info.get("size_mb"),
                    "notes": version_info.get("notes", ""),
                    "hash": version_info.get("hash"),
                    "current_version": self.current_version,
                }
            else:
                return {
                    "update_available": False,
                    "current_version": self.current_version,
                    "latest_version": version_info.get("version"),
                }
        except Exception as e:
            return {
                "update_available": False,
                "error": str(e),
                "current_version": self.current_version,
            }

    def download_update(
        self, download_url: str, expected_hash: str | None = None
    ) -> tuple[bool, str]:
        """
        Download update package and verify integrity.

        Args:
            download_url: URL to download the update package
            expected_hash: Expected SHA-256 hash for verification

        Returns:
            Tuple of (success, file_path or error_message)
        """
        try:
            download_url = _require_https_url(download_url, "download_url")
            temp_dir = tempfile.mkdtemp(prefix="distribai_update_")
            filename = _safe_download_filename(download_url)
            file_path = os.path.join(temp_dir, filename)

            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    print(
                        f"\rDownloading: {percent}% ({block_num * block_size}/{total_size} bytes)",
                        end="",
                    )

            urllib.request.urlretrieve(download_url, file_path, progress_hook)
            print()  # New line after progress

            if expected_hash:
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)

                actual_hash = sha256_hash.hexdigest()
                if actual_hash != expected_hash:
                    os.remove(file_path)
                    return (
                        False,
                        f"Hash verification failed: expected {expected_hash}, got {actual_hash}",
                    )

            if not os.path.exists(file_path):
                return False, "Download failed - file not created"

            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                return False, "Download failed - empty file"

            return True, file_path

        except urllib.error.URLError as e:
            return False, f"Network error: {e}"
        except urllib.error.HTTPError as e:
            return False, f"HTTP error: {e.code} - {e.reason}"
        except OSError as e:
            return False, f"File system error: {e}"
        except Exception as e:
            return False, f"Unexpected download error: {e}"

    def verify_update_package(self, file_path: str, signature_url: str | None = None) -> bool:
        """
        Verify the integrity and authenticity of an update package.

        When a release-signing key is configured (``DISTRIBAI_UPDATE_PUBLIC_KEY``)
        the package's detached Ed25519 signature is cryptographically verified
        against the file bytes — a missing signature URL or a bad signature is
        rejected. Without a configured key, verification falls back to the
        historical best-effort checks (file exists, non-trivial size, and a
        well-formed signature when one is supplied) and logs that signatures
        are not being enforced.

        Args:
            file_path: Path to the downloaded update package.
            signature_url: HTTPS URL of the detached signature (hex/base64).

        Returns:
            True if the package is verified, False otherwise.
        """
        try:
            if not os.path.exists(file_path):
                return False

            file_size = os.path.getsize(file_path)
            if file_size < 1024:  # Less than 1KB seems suspicious
                return False

            public_key = _load_update_public_key()

            if public_key is not None:
                # Signature enforcement is on: a signature is mandatory.
                if not signature_url:
                    logger.error(
                        "Update signing key configured but no signature_url provided; "
                        "refusing unsigned package"
                    )
                    return False
                signature = self._fetch_signature(signature_url)
                if signature is None:
                    return False
                with open(file_path, "rb") as handle:
                    package_bytes = handle.read()
                try:
                    public_key.verify(signature, package_bytes)
                except InvalidSignature:
                    logger.error("Update package signature verification FAILED")
                    return False
                logger.info("Update package Ed25519 signature verified")
                return True

            # No signing key configured: best-effort integrity only.
            if signature_url:
                signature = self._fetch_signature(signature_url)
                if signature is None:
                    return False
            logger.warning(
                "%s is not set; update signatures are NOT cryptographically enforced. "
                "Set it to require signed releases.",
                UPDATE_PUBLIC_KEY_ENV,
            )
            return True

        except OSError as exc:
            logger.error("Failed to read update package for verification: %s", exc)
            return False

    def _fetch_signature(self, signature_url: str) -> bytes | None:
        """Download and decode a detached signature from an HTTPS URL."""
        try:
            signature_url = _require_https_url(signature_url, "signature_url")
        except ValueError as exc:
            logger.error("Rejecting signature URL: %s", exc)
            return None
        try:
            with urllib.request.urlopen(signature_url, timeout=10) as response:
                signature_text = response.read().decode("utf-8")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.error("Failed to download signature: %s", exc)
            return None
        signature = _decode_signature(signature_text)
        if signature is None:
            logger.error("Signature is not a valid 64-byte hex/base64 Ed25519 signature")
        return signature

    def install_update(self, package_path: str, backup_dir: str | None = None) -> tuple[bool, str]:
        """
        Install the update package.

        Supports:
        - ``.whl`` via ``pip install --upgrade``
        - ``.zip`` / ``.tar.gz`` by unpacking and merging top-level entries into
          ``DISTRIBAI_INSTALL_ROOT`` (defaults to repository root containing ``services_python``).
        """
        path = Path(package_path)
        if not path.exists():
            return False, "Update package not found"

        install_root = Path(
            os.getenv("DISTRIBAI_INSTALL_ROOT", Path(__file__).resolve().parent.parent)
        )
        name_lower = path.name.lower()

        if name_lower.endswith(".whl"):
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if proc.returncode != 0:
                    tail = (proc.stderr or proc.stdout or "").strip()
                    return False, tail[-2000:] if tail else "pip install failed"
                return True, "Python wheel installed via pip; restart node services if needed."
            except subprocess.TimeoutExpired:
                return False, "pip install timed out"
            except Exception as exc:
                return False, f"Wheel install failed: {exc}"

        try:
            with tempfile.TemporaryDirectory(prefix="distribai_update_unpack_") as extract_dir:
                extract_root = Path(extract_dir)
                try:
                    _safe_unpack_archive(path, extract_root)
                except ValueError as exc:
                    return False, str(exc)

                entries = list(extract_root.iterdir())
                if not entries:
                    return False, "Archive was empty"

                if backup_dir:
                    backup_path = Path(backup_dir) / f"backup_{int(time.time())}"
                    shutil.copytree(
                        install_root,
                        backup_path,
                        ignore=shutil.ignore_patterns("*.pyc", "__pycache__", ".git"),
                        dirs_exist_ok=True,
                    )

                for src in entries:
                    dest = install_root / src.name
                    if src.is_dir():
                        shutil.copytree(src, dest, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dest)

            return True, f"Update merged into {install_root}; restart recommended."

        except Exception as exc:
            return False, f"Installation failed: {exc}"

    def _is_newer_version(self, latest: str, current: str) -> bool:
        """
        Compare version strings to check if latest is newer than current.

        Args:
            latest: Latest version string
            current: Current version string

        Returns:
            True if latest is newer than current
        """
        try:
            latest_parts = [int(x) for x in latest.split(".")]
            current_parts = [int(x) for x in current.split(".")]

            # Pad with zeros to make same length
            max_len = max(len(latest_parts), len(current_parts))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            current_parts.extend([0] * (max_len - len(current_parts)))

            return latest_parts > current_parts
        except (ValueError, AttributeError):
            return False

    def _calculate_sha256(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA-256 hash as hexadecimal string
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


def create_version_json(version: str, download_url: str, file_path: str, notes: str = "") -> dict:
    """
    Create version.json content for a release.

    Args:
        version: Version string (e.g., "1.0.0")
        download_url: Base URL for downloads
        file_path: Path to the built executable
        notes: Release notes

    Returns:
        Dictionary suitable for version.json
    """
    # Calculate file hash
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    # Get file size in MB
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    return {
        "version": version,
        "download_url": download_url,
        "size_mb": round(file_size_mb, 2),
        "notes": notes,
        "hash": sha256_hash.hexdigest(),
        "release_date": time.strftime("%Y-%m-%d"),
        "platform": "windows" if os.name == "nt" else "linux" if os.name == "posix" else "macos",
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the update service
    update_service = UpdateService(
        "https://github.com/test-org/distribai-releases/releases/latest/download"
    )

    # Check for updates
    update_info = update_service.check_for_updates()
    print("Update check result:", json.dumps(update_info, indent=2))

    # Example of creating version.json
    if len(sys.argv) > 1:
        version_info = create_version_json(
            version="1.0.1",
            download_url="https://github.com/test-org/distribai-releases/releases/download/v1.0.1/",
            file_path=sys.argv[1],
            notes="Bug fixes and performance improvements",
        )
        print("Version JSON:", json.dumps(version_info, indent=2))
