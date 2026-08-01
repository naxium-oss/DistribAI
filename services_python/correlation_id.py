"""Cross-service request correlation IDs for the DistribAI admin HTTP API.

Propagates ``X-Request-Id`` (alias ``X-Correlation-Id``) through aiohttp
middleware into request state, response headers, and structured log records.
"""

from __future__ import annotations

import contextvars
import logging
import secrets
from typing import Final

from aiohttp import web

REQUEST_ID_HEADER: Final[str] = "X-Request-Id"
CORRELATION_ID_HEADER: Final[str] = "X-Correlation-Id"
# Instantiated at import so aiohttp sees a module-level caller frame.
CorrelationIdAppKey = web.RequestKey("correlation_id", str)

_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "distribai_correlation_id", default=""
)


def get_correlation_id() -> str:
    """Return the correlation id bound to the current context (may be empty)."""
    return _correlation_id.get() or ""


def set_correlation_id(value: str) -> contextvars.Token[str]:
    """Bind ``value`` as the active correlation id; return a reset token."""
    return _correlation_id.set(value or "")


def generate_correlation_id() -> str:
    """Mint a new opaque correlation / request id."""
    return secrets.token_urlsafe(16)


def resolve_incoming_correlation_id(request: web.Request) -> str:
    """Prefer client ``X-Request-Id``, then ``X-Correlation-Id``, else mint one."""
    for header in (REQUEST_ID_HEADER, CORRELATION_ID_HEADER):
        raw = (request.headers.get(header) or "").strip()
        if raw and len(raw) <= 128:
            # Keep printable / URL-safe-ish ids; strip control characters.
            cleaned = "".join(ch for ch in raw if ch.isprintable() and ch not in "\r\n")
            if cleaned:
                return cleaned[:128]
    return generate_correlation_id()


class CorrelationIdFilter(logging.Filter):
    """Inject ``correlation_id`` onto every log record for structured logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True


def ensure_correlation_logging(root: logging.Logger | None = None) -> None:
    """Attach :class:`CorrelationIdFilter` once to the given (or root) logger."""
    target = root if root is not None else logging.getLogger()
    for existing in target.filters:
        if isinstance(existing, CorrelationIdFilter):
            return
    target.addFilter(CorrelationIdFilter())


@web.middleware
async def correlation_id_middleware(
    request: web.Request, handler
) -> web.StreamResponse:
    """Generate/propagate correlation ids and echo them on the response."""
    correlation_id = resolve_incoming_correlation_id(request)
    request[CorrelationIdAppKey] = correlation_id
    token = set_correlation_id(correlation_id)
    try:
        logger = logging.getLogger("services_python.admin")
        logger.debug(
            "admin_request method=%s path=%s correlation_id=%s",
            request.method,
            request.path,
            correlation_id,
        )
        response = await handler(request)
    finally:
        _correlation_id.reset(token)

    response.headers[REQUEST_ID_HEADER] = correlation_id
    response.headers[CORRELATION_ID_HEADER] = correlation_id
    return response
