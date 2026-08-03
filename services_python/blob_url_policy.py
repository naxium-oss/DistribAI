"""Host and path allowlisting for gradient / blob URLs fetched by the orchestrator."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse


def allowed_blob_hosts() -> frozenset[str]:
    raw = os.getenv("ALLOWED_BLOB_HOSTS", "127.0.0.1,localhost")
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def allowed_s3_buckets() -> frozenset[str]:
    buckets: set[str] = set()
    primary = os.getenv("S3_BUCKET_NAME", "").strip()
    if primary:
        buckets.add(primary)
    extra = os.getenv("ALLOWED_S3_BUCKETS", "")
    for part in extra.split(","):
        part = part.strip()
        if part:
            buckets.add(part)
    return frozenset(buckets)


def _allowed_local_roots() -> list[Path]:
    """Resolved directories under which file:// and bare paths may resolve."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        (repo_root / "runtime").resolve(),
        Path(os.getenv("GRADIENT_LOCAL_ROOT", str(repo_root / "runtime"))).resolve(),
    ]
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in candidates:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def _is_under_runtime_roots(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in _allowed_local_roots():
        if resolved == root or root in resolved.parents:
            return True
    return False


def sanitize_s3_object_key(key: str) -> str | None:
    """Collapse an S3 object key; return None for empty keys or path traversal."""
    if not key or not isinstance(key, str):
        return None
    normalized = key.strip().replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        return None
    parts = [part for part in normalized.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        return None
    return "/".join(parts)


def is_allowed_gradient_url(url: str) -> bool:
    """True when the orchestrator is permitted to download a gradient blob from ``url``."""
    if not url or not url.strip():
        return False

    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()

    if scheme in ("http", "https"):
        host = (parsed.hostname or "").lower()
        if not host:
            return False
        if parsed.username or parsed.password:
            return False
        return host in allowed_blob_hosts()

    if scheme == "s3":
        bucket = (parsed.netloc or "").strip()
        key = sanitize_s3_object_key(parsed.path.lstrip("/"))
        if not bucket or not key:
            return False
        allowed = allowed_s3_buckets()
        return bool(allowed) and bucket in allowed

    if scheme == "file":
        path = Path(parsed.path)
        if not path.exists() and parsed.netloc:
            path = Path(f"{parsed.netloc}{parsed.path}")
        return _is_under_runtime_roots(path)

    if scheme == "":
        if _is_under_runtime_roots(Path(url)):
            return True
        # A Windows-style path (backslash separators) evaluated on a POSIX
        # host — or vice versa — is not split into components by the native
        # Path. Retry with normalized separators so cross-platform bare paths
        # under an allowed root are still accepted.
        if "\\" in url:
            return _is_under_runtime_roots(Path(url.replace("\\", "/")))
        return False

    # Windows drive-letter paths show up as a one-character "scheme".
    if len(scheme) == 1 and len(url) >= 3 and url[1:3] in (":\\", ":/"):
        if _is_under_runtime_roots(Path(url)):
            return True
        return _is_under_runtime_roots(Path(url.replace("\\", "/")))

    return False
