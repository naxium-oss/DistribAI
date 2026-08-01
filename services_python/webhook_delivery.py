"""Signed HTTP webhook delivery for DistribAI job terminal callbacks.

When a job includes ``hparams.callback_url``, the orchestrator POSTs a signed
JSON payload to that URL on terminal job states (success, failed, cancelled,
timeout, error, completed).

Signature header: ``X-DistribAI-Signature: sha256=<hex>`` using ``SIGNING_KEY``.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import logging
import os
import socket
import threading
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from services_python.constants import SIGNING_KEY
from services_python.env_bool import env_truthy

logger = logging.getLogger(__name__)

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"success", "failed", "cancelled", "timeout", "error", "completed"}
)
SIGNATURE_HEADER: str = "X-DistribAI-Signature"
CONTENT_TYPE: str = "application/json"
DEFAULT_TIMEOUT_SECONDS: float = 10.0
_MAX_CALLBACK_URL_LEN: int = 2048


def is_terminal_status(status: str | None) -> bool:
    """Return True when ``status`` is a job terminal state."""
    return str(status or "").strip().lower() in TERMINAL_STATUSES


def sign_payload(body: bytes, signing_key: str | None = None) -> str:
    """Return ``sha256=<hex>`` HMAC of ``body`` using SIGNING_KEY."""
    key = (signing_key if signing_key is not None else SIGNING_KEY).encode("utf-8")
    digest = hmac.new(key, body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(body: bytes, header_value: str, signing_key: str | None = None) -> bool:
    """Constant-time check that ``header_value`` matches ``sign_payload``."""
    expected = sign_payload(body, signing_key=signing_key)
    return hmac.compare_digest(expected, (header_value or "").strip())


def _host_is_loopback(hostname: str) -> bool:
    host = (hostname or "").strip().lower().rstrip(".")
    if not host:
        return True
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        raw = info[4][0]
        try:
            if ipaddress.ip_address(raw).is_loopback:
                return True
        except ValueError:
            continue
    return False


def callback_url_allowed(url: str | None) -> bool:
    """Accept absolute http(s) URLs; block loopback unless explicitly allowed."""
    if not url or not isinstance(url, str):
        return False
    cleaned = url.strip()
    if not cleaned or len(cleaned) > _MAX_CALLBACK_URL_LEN:
        return False
    parsed = urlparse(cleaned)
    if parsed.scheme not in ("http", "https"):
        return False
    if not parsed.hostname:
        return False
    allow_loopback = env_truthy("DISTRIBAI_ALLOW_LOOPBACK_WEBHOOKS") is True
    if not allow_loopback and _host_is_loopback(parsed.hostname):
        return False
    return True


def validate_callback_url(url: str | None) -> str | None:
    """Return cleaned URL when allowed, else None."""
    if not url or not isinstance(url, str):
        return None
    cleaned = url.strip()
    if callback_url_allowed(cleaned):
        return cleaned
    return None


def extract_callback_url(hparams: dict[str, Any] | None) -> str | None:
    """Read and validate ``callback_url`` from job/task hparams."""
    if not isinstance(hparams, dict):
        return None
    return validate_callback_url(hparams.get("callback_url"))


def build_webhook_payload(
    job: dict[str, Any] | str,
    status: str = "",
    reason: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build the JSON body posted to ``callback_url``.

    Compatible call shapes:
    - ``build_webhook_payload(job_dict, status, reason)``
    - ``build_webhook_payload(job_id=..., status=..., ...)``
    """
    if isinstance(job, str) and not status and "status" in kwargs:
        # Keyword-style: first positional was job_id string by mistake — rare.
        job_id = job
        job = {"job_id": job_id}

    if not isinstance(job, dict):
        job = {"job_id": str(job)}

    if kwargs.get("job_id") and "job_id" not in job:
        job = {**job, "job_id": kwargs["job_id"]}

    status_value = status or str(kwargs.get("status") or job.get("status") or "")
    reason_value = reason or str(kwargs.get("reason") or job.get("latest_reason") or "")

    payload: dict[str, Any] = {
        "event": "job.terminal",
        "job_id": str(job.get("job_id") or kwargs.get("job_id") or ""),
        "status": status_value,
        "reason": reason_value,
        "model_name": str(kwargs.get("model_name") or job.get("model_name") or ""),
        "priority_tier": str(kwargs.get("priority_tier") or job.get("priority_tier") or ""),
        "progress_pct": float(kwargs.get("progress_pct", job.get("progress_pct") or 0.0)),
        "source": "DistribAI",
    }
    return payload


def _hparams_from_job(job: dict[str, Any]) -> dict[str, Any]:
    for key in ("hparams", "hyperparams"):
        raw = job.get(key)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
    for task in job.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        for key in ("hparams", "hyperparams", "hparams_json"):
            raw = task.get(key)
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
    return {}


def get_job_hparams(db: Any, job_id: str) -> dict[str, Any]:
    """Load hparams JSON for a job from DB helper or job/task rows."""
    getter = getattr(db, "get_job_hparams", None)
    if callable(getter):
        raw = getter(job_id)
        return dict(raw) if isinstance(raw, dict) else {}
    job = db.get_job(job_id) if hasattr(db, "get_job") else None
    if isinstance(job, dict):
        return _hparams_from_job(job)
    return {}


def deliver_webhook(
    callback_url: str,
    payload: dict[str, Any],
    *,
    signing_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bool, int | None, str]:
    """POST signed JSON synchronously. Returns ``(ok, status_code, detail)``."""
    url = validate_callback_url(callback_url)
    if not url:
        return False, None, "invalid callback_url"

    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    headers = {
        "Content-Type": CONTENT_TYPE,
        "User-Agent": "DistribAI-Webhook/1.0",
        SIGNATURE_HEADER: sign_payload(body, signing_key=signing_key),
        "X-DistribAI-Event": str(payload.get("event") or "job.terminal"),
        "X-DistribAI-Job-Id": str(payload.get("job_id") or ""),
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            code = int(getattr(response, "status", 200) or 200)
            return 200 <= code < 300, code, "delivered"
    except urllib.error.HTTPError as exc:
        logger.warning(
            "Webhook delivery HTTP error job_id=%s url=%s status=%s",
            payload.get("job_id"),
            url,
            exc.code,
        )
        return False, int(exc.code), str(exc.reason or "http_error")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning(
            "Webhook delivery failed job_id=%s url=%s err=%s",
            payload.get("job_id"),
            url,
            exc,
        )
        return False, None, str(exc)


def schedule_webhook_delivery(
    callback_url: str,
    payload: dict[str, Any],
    *,
    signing_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Fire-and-forget delivery on a daemon thread (safe from sync DB paths)."""

    def _run() -> None:
        ok, code, detail = deliver_webhook(
            callback_url,
            payload,
            signing_key=signing_key,
            timeout=timeout,
        )
        if ok:
            logger.info(
                "Webhook delivered job_id=%s status=%s http=%s",
                payload.get("job_id"),
                payload.get("status"),
                code,
            )
        else:
            logger.warning(
                "Webhook not delivered job_id=%s detail=%s http=%s",
                payload.get("job_id"),
                detail,
                code,
            )

    thread = threading.Thread(
        target=_run,
        name=f"distribai-webhook-{payload.get('job_id', 'unknown')}",
        daemon=True,
    )
    thread.start()


def schedule_job_callback(
    job: dict[str, Any],
    status: str,
    reason: str = "",
    *,
    sync: bool = False,
    signing_key: str | None = None,
) -> bool:
    """Schedule (or sync-deliver) a signed webhook for a terminal job row.

    Expects ``job`` to include ``job_id`` and preferably ``hparams`` /
    ``hyperparams`` or task ``hparams_json`` with ``callback_url``.
    """
    if not isinstance(job, dict):
        return False
    job_id = str(job.get("job_id") or "")
    if not job_id or not is_terminal_status(status):
        return False

    hparams = _hparams_from_job(job)
    callback_url = extract_callback_url(hparams)
    if not callback_url:
        return False

    payload = build_webhook_payload(job, status, reason)
    if sync or os.getenv("DISTRIBAI_WEBHOOK_SYNC", "").lower() in {"1", "true", "yes"}:
        ok, _code, _detail = deliver_webhook(
            callback_url, payload, signing_key=signing_key
        )
        return ok

    schedule_webhook_delivery(callback_url, payload, signing_key=signing_key)
    return True


def notify_job_terminal(
    db: Any,
    job_id: str,
    status: str,
    reason: str = "",
    *,
    sync: bool = False,
    signing_key: str | None = None,
) -> bool:
    """Load the job from ``db`` and schedule a signed terminal webhook."""
    if not job_id or not is_terminal_status(status):
        return False
    job: dict[str, Any] = {"job_id": job_id}
    if hasattr(db, "get_job"):
        loaded = db.get_job(job_id)
        if isinstance(loaded, dict):
            job = loaded
    # Attach hparams from tasks when the job row lacks them.
    if not _hparams_from_job(job):
        hparams = get_job_hparams(db, job_id)
        if hparams:
            job = {**job, "hparams": hparams}
    return schedule_job_callback(
        job, status, reason, sync=sync, signing_key=signing_key
    )
