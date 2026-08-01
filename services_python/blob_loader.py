"""Canonical async blob loaders (allowlist enforced)."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

from services_python.blob_url_policy import is_allowed_gradient_url, sanitize_s3_object_key

logger = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT_SEC = 30.0
DEFAULT_MAX_TEXT_BYTES = 10 * 1024 * 1024


def _resolve_local_path(blob_url: str, parsed) -> Path:
    if parsed.scheme == "file":
        path = Path(parsed.path)
        if not path.exists() and parsed.netloc:
            path = Path(f"{parsed.netloc}{parsed.path}")
        return path
    return Path(parsed.path if parsed.scheme == "" else blob_url)


async def load_text_blob(
    blob_url: str,
    *,
    http_timeout: float = DEFAULT_HTTP_TIMEOUT_SEC,
    max_bytes: int = DEFAULT_MAX_TEXT_BYTES,
) -> str | None:
    """Load UTF-8 text from file or http(s) URL after allowlist check."""
    if not is_allowed_gradient_url(blob_url):
        logger.warning("Blocked blob URL outside allowlist: %s", blob_url)
        return None

    parsed = urlparse(blob_url)
    scheme = (parsed.scheme or "").lower()

    try:
        if scheme in ("", "file") or (len(scheme) == 1 and len(blob_url) >= 3 and blob_url[1:3] in (":\\", ":/")):
            path = _resolve_local_path(blob_url, parsed)
            data = path.read_text(encoding="utf-8")
            if len(data.encode("utf-8")) > max_bytes:
                logger.warning("Blob file too large: %s", blob_url)
                return None
            return data

        if scheme in ("http", "https"):
            timeout = aiohttp.ClientTimeout(total=http_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(blob_url) as resp:
                    if resp.status >= 400:
                        logger.warning("HTTP blob fetch failed: %s status=%s", blob_url, resp.status)
                        return None
                    content = await resp.text()
                    if len(content.encode("utf-8")) > max_bytes:
                        logger.warning("HTTP blob payload too large: %s", blob_url)
                        return None
                    return content
    except OSError as exc:
        logger.warning("Blob file IO error for %s: %s", blob_url, exc)
        return None
    except aiohttp.ClientError:
        logger.exception("Failed to load text blob from %s", blob_url)
        return None

    logger.warning("Unsupported text blob URL scheme: %s", blob_url)
    return None


async def load_json_blob(
    blob_url: str,
    *,
    http_timeout: float = DEFAULT_HTTP_TIMEOUT_SEC,
    s3_client: Any | None = None,
) -> dict[str, Any] | None:
    """Load JSON from file, http(s), or s3 URL after allowlist check."""
    if not is_allowed_gradient_url(blob_url):
        logger.warning("Blocked blob URL outside allowlist: %s", blob_url)
        return None

    parsed = urlparse(blob_url)
    scheme = (parsed.scheme or "").lower()

    try:
        if scheme in ("", "file") or (len(scheme) == 1 and len(blob_url) >= 3 and blob_url[1:3] in (":\\", ":/")):
            path = _resolve_local_path(blob_url, parsed)
            return json.loads(path.read_text(encoding="utf-8"))

        if scheme in ("http", "https"):
            timeout = aiohttp.ClientTimeout(total=http_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(blob_url) as resp:
                    if resp.status >= 400:
                        logger.warning("HTTP blob fetch failed: %s status=%s", blob_url, resp.status)
                        return None
                    return await resp.json()

        if scheme == "s3":
            bucket = parsed.netloc
            key = sanitize_s3_object_key(parsed.path.lstrip("/"))
            if not bucket or not key:
                logger.warning("Invalid S3 blob URL: %s", blob_url)
                return None
            if not s3_client:
                logger.warning("S3 blob load requested but S3 is not configured: %s", blob_url)
                return None

            def _download() -> dict[str, Any]:
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                body = obj["Body"].read()
                return json.loads(body.decode("utf-8"))

            return await asyncio.to_thread(_download)
    except json.JSONDecodeError as exc:
        logger.warning("Blob payload is not valid JSON for %s: %s", blob_url, exc)
        return None
    except OSError as exc:
        logger.warning("Blob file/S3 IO error for %s: %s", blob_url, exc)
        return None
    except (TypeError, ValueError, aiohttp.ClientError):
        logger.exception("Failed to load blob from %s", blob_url)
        return None

    logger.warning("Unsupported blob URL scheme: %s", blob_url)
    return None
