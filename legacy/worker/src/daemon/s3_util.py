"""
S3 Utility for DistribAI Worker

Production-grade S3 utility using aioboto3 for asynchronous blob transfers.
Handles weight downloads and gradient uploads.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import aioboto3
import aiofiles
import aiohttp
from dotenv import load_dotenv

from services_python.blob_url_policy import is_allowed_gradient_url, sanitize_s3_object_key
from services_python.harness_policy import harness_disables_s3

load_dotenv()

logger = logging.getLogger(__name__)


class S3Manager:
    """
    Manages S3 operations for the DistribAI worker.

    Handles asynchronous file uploads and downloads to/from S3 or
    compatible object storage. Supports both S3 URIs and pre-signed URLs.

    Attributes:
        access_key: AWS access key ID
        secret_key: AWS secret access key
        region: AWS region
        bucket_name: S3 bucket name
        session: aioboto3 session for async operations

    Example:
        s3 = S3Manager()
        success = await s3.download_file("s3://bucket/key", "/local/path")
        url = await s3.upload_file("/local/file", "key")
    """

    def __init__(self):
        """
        Initialize the S3 manager.

        Loads credentials from environment variables:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_REGION (default: us-east-1)
        - S3_BUCKET_NAME

        Example:
            >>> s3 = S3Manager()
        """
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        if harness_disables_s3():
            self.bucket_name = None
            self.access_key = None
            self.secret_key = None
        self.session = aioboto3.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region,
        )

    async def download_file(self, s3_url: str, local_path: str) -> bool:
        """
        Download a file from S3 or local storage.

        Handles both s3:// paths and pre-signed URLs. If s3_url starts
        with http, it is treated as a pre-signed URL. If it's a local
        file path, it copies from local to local.

        Args:
            s3_url: S3 URL, pre-signed URL, or local file path
            local_path: Local destination path

        Returns:
            True if download successful, False otherwise

        Example:
            >>> success = await s3.download_file(
            ...     s3_url="s3://bucket/model.pt",
            ...     local_path="/local/model.pt"
            ... )
        """
        logger.info(f"Downloading from S3: {s3_url} -> {local_path}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not is_allowed_gradient_url(s3_url):
            logger.error("Blocked download from disallowed blob URL: %s", s3_url)
            return False
        try:
            bare_local = Path(s3_url)
            if bare_local.is_file():
                async with (
                    aiofiles.open(bare_local, "rb") as source_file,
                    aiofiles.open(local_path, "wb") as target_file,
                ):
                    await target_file.write(await source_file.read())
                return True
            parsed = urlparse(s3_url)
            if parsed.scheme in ("", "file"):
                source = Path(parsed.path if parsed.scheme == "file" else s3_url)
                if not source.exists():
                    logger.error(f"Local blob does not exist: {source}")
                    return False
                async with (
                    aiofiles.open(source, "rb") as source_file,
                    aiofiles.open(local_path, "wb") as target_file,
                ):
                    await target_file.write(await source_file.read())
                return True
            if parsed.scheme in {"http", "https"}:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as session:
                    async with session.get(s3_url) as response:
                        if response.status == 200:
                            async with aiofiles.open(local_path, "wb") as f:
                                await f.write(await response.read())
                            return True
                        logger.error(f"Failed to download URL: {response.status}")
                        return False
            else:
                bucket, key = self._parse_s3_url(s3_url)
                async with self.session.client("s3") as s3:
                    await s3.download_file(bucket, key, local_path)
                return True
        except Exception as e:
            logger.error(f"S3 Download Error: {e}", exc_info=True)
            return False

    async def upload_file(self, local_path: str, s3_key: str) -> str | None:
        """
        Upload a file to S3.

        Args:
            local_path: Local file path to upload
            s3_key: S3 object key

        Returns:
            S3 URL (s3://bucket/key) if successful, local path if S3 not configured,
            None if upload fails

        Example:
            >>> url = await s3.upload_file("/local/file.pt", "gradients/file.pt")
            >>> print(f"Uploaded to: {url}")
        """
        if not self.bucket_name:
            logger.warning("S3_BUCKET_NAME not configured. Using local file path for blob handoff.")
            local_target = Path(local_path).resolve()
            if not is_allowed_gradient_url(str(local_target)):
                logger.error("Blocked upload handoff to disallowed local path: %s", local_target)
                return None
            return str(local_target)
        safe_key = sanitize_s3_object_key(s3_key)
        if not safe_key:
            logger.error("Blocked upload with invalid S3 key: %s", s3_key)
            return None
        candidate_url = f"s3://{self.bucket_name}/{safe_key}"
        if not is_allowed_gradient_url(candidate_url):
            logger.error("Blocked upload to disallowed S3 destination: %s", candidate_url)
            return None
        logger.info(f"Uploading to S3: {local_path} -> {candidate_url}")
        try:
            async with self.session.client("s3") as s3:
                await s3.upload_file(local_path, self.bucket_name, safe_key)
            return f"s3://{self.bucket_name}/{safe_key}"
        except Exception as e:
            logger.error(f"S3 Upload Error: {e}", exc_info=True)
            fallback = str(Path(local_path).resolve())
            if is_allowed_gradient_url(fallback):
                logger.warning(
                    "Falling back to local file path for blob handoff after S3 upload failure."
                )
                return fallback
            logger.error("Blocked local fallback path after upload failure: %s", fallback)
            return None

    def _parse_s3_url(self, s3_url: str) -> tuple[str, str]:
        """
        Parse an S3 URL into bucket and key components.

        Args:
            s3_url: S3 URL in format s3://bucket/key

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If URL is invalid or contains path traversal attempts

        Example:
            >>> bucket, key = s3._parse_s3_url("s3://my-bucket/path/to/file.txt")
            >>> print(f"Bucket: {bucket}, Key: {key}")
        """
        if not s3_url or not isinstance(s3_url, str):
            raise ValueError(f"Invalid S3 URL: {s3_url}")
        if not s3_url.startswith("s3://"):
            raise ValueError(f"Invalid S3 URL: {s3_url}")
        if ".." in s3_url or "~" in s3_url:
            raise ValueError(f"[SECURITY] Path traversal detected in S3 URL: {s3_url}")
        if "\x00" in s3_url or any(ord(c) < 32 for c in s3_url):
            raise ValueError(f"[SECURITY] Invalid characters in S3 URL: {s3_url}")
        parts = s3_url[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        bucket, key = parts[0], parts[1]
        if not bucket or len(bucket) < 3 or len(bucket) > 63:
            raise ValueError(f"Invalid S3 bucket name: {bucket}")
        if not all(c.isalnum() or c in "-." for c in bucket):
            raise ValueError(f"Invalid S3 bucket name characters: {bucket}")
        if key.startswith("/") or "//" in key:
            raise ValueError(f"Invalid S3 key format: {key}")
        return bucket, key

    def _is_s3_url(self, url: str) -> bool:
        """
        Check if a URL is an S3 URL.

        Args:
            url: URL to check

        Returns:
            True if URL starts with s3://, False otherwise

        Example:
            >>> is_s3 = s3._is_s3_url("s3://bucket/key")
            >>> print(f"Is S3: {is_s3}")
        """
        if not url or not isinstance(url, str):
            return False
        return url.startswith("s3://")

    async def health_check(self) -> bool:
        """
        Check S3 connectivity by attempting to access the bucket.

        Returns:
            True if bucket is accessible, False otherwise

        Example:
            >>> if await s3.health_check():
            ...     print("S3 is accessible")
        """
        if not self.bucket_name:
            logger.error("S3_BUCKET_NAME not configured")
            return False
        try:
            async with self.session.client("s3") as s3:
                await s3.head_bucket(Bucket=self.bucket_name)
            return True
        except Exception as e:
            logger.error(f"S3 health check failed: {e}")
            return False
