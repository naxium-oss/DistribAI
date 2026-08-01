"""On-disk and optional S3 storage for job script tarballs."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

from services_python.harness_policy import harness_disables_s3

logger = logging.getLogger(__name__)

_TASK_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
_DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "bundles"
MAX_BUNDLE_BYTES = 5_000_000
_S3_PREFIX = "bundles"


def bundle_root() -> Path:
    raw = os.getenv("DISTRIBAI_BUNDLE_DIR", "").strip()
    return Path(raw) if raw else _DEFAULT_ROOT


def _bundle_path(task_id: str) -> Path:
    if not _TASK_ID_RE.match(task_id):
        raise ValueError("invalid task_id for bundle storage")
    return bundle_root() / f"{task_id}.tar.gz"


def _s3_key(task_id: str) -> str:
    return f"{_S3_PREFIX}/{task_id}.tar.gz"


def _s3_client_and_bucket():
    if harness_disables_s3():
        return None, None
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket:
        return None, None
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            config=Config(signature_version="s3v4"),
        )
        return client, bucket
    except Exception as exc:
        logger.warning("S3 bundle client unavailable: %s", exc)
        return None, None


def _upload_s3(task_id: str, data: bytes) -> None:
    client, bucket = _s3_client_and_bucket()
    if not client or not bucket:
        return
    try:
        client.put_object(
            Bucket=bucket,
            Key=_s3_key(task_id),
            Body=data,
            ContentType="application/gzip",
        )
    except Exception as exc:
        logger.warning("S3 bundle upload failed for %s: %s", task_id, exc)


def _download_s3(task_id: str) -> bytes | None:
    client, bucket = _s3_client_and_bucket()
    if not client or not bucket:
        return None
    try:
        response = client.get_object(Bucket=bucket, Key=_s3_key(task_id))
        body = response["Body"].read()
        if len(body) > MAX_BUNDLE_BYTES:
            logger.warning("S3 bundle for %s exceeds size cap", task_id)
            return None
        return body
    except Exception as exc:
        from botocore.exceptions import ClientError

        if isinstance(exc, ClientError) and exc.response.get("Error", {}).get("Code") == "NoSuchKey":
            return None
        logger.warning("S3 bundle download failed for %s: %s", task_id, exc)
        return None


def save_bundle(task_id: str, data: bytes) -> Path:
    """Write tarball bytes for a task; mirrors to S3 when configured."""
    if not data:
        raise ValueError("empty script bundle")
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("script bundle too large")
    path = _bundle_path(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    _upload_s3(task_id, data)
    return path


def load_bundle(task_id: str) -> bytes | None:
    """Read tarball bytes from disk, then S3 if missing locally."""
    try:
        path = _bundle_path(task_id)
    except ValueError:
        return None
    if path.is_file():
        size = path.stat().st_size
        if size <= MAX_BUNDLE_BYTES:
            return path.read_bytes()
    return _download_s3(task_id)


def delete_bundle(task_id: str) -> bool:
    """Remove stored bundle from disk and S3; returns True if disk file was removed."""
    removed = False
    try:
        path = _bundle_path(task_id)
        if path.is_file():
            path.unlink()
            removed = True
    except ValueError:
        pass
    client, bucket = _s3_client_and_bucket()
    if client and bucket:
        try:
            client.delete_object(Bucket=bucket, Key=_s3_key(task_id))
        except Exception as exc:
            logger.warning("S3 bundle delete failed for %s: %s", task_id, exc)
    return removed
