"""Admin HTTP API authentication middleware."""

from __future__ import annotations

import hmac
import logging
import os
from typing import Final

from aiohttp import web

from services_python.constants import DEFAULT_ADMIN_HOST
from services_python.env_bool import env_truthy
from services_python.network_policy import is_loopback_host

logger = logging.getLogger(__name__)

_ADMIN_PATH_PREFIXES: Final[tuple[str, ...]] = ("/admin/", "/api/admin/", "/api/docs/")
# Only liveness stays open: surge triggering was previously exempt too, which
# let any unauthenticated caller fire a global credit surge.
_EXEMPT_PATHS: Final[frozenset[str]] = frozenset({"/admin/health"})


def admin_auth_enforced() -> bool:
    """Return whether admin routes require a shared secret."""
    explicit = env_truthy("ADMIN_REQUIRE_AUTH")
    if explicit is not None:
        return explicit

    host = os.getenv("ADMIN_HOST", DEFAULT_ADMIN_HOST).strip().lower()
    return not is_loopback_host(host)


def resolve_admin_secret() -> str | None:
    """Secret used to authorize admin HTTP calls when enforcement is on."""
    dedicated = os.getenv("DISTRIBAI_ADMIN_SECRET", "").strip()
    if dedicated:
        return dedicated
    if not admin_auth_enforced():
        return None
    return None


def _path_requires_admin_auth(path: str) -> bool:
    if path in _EXEMPT_PATHS:
        return False
    return any(path.startswith(prefix) for prefix in _ADMIN_PATH_PREFIXES)


def _extract_bearer_token(request: web.Request) -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    header_token = request.headers.get("X-Admin-Token", "").strip()
    return header_token or None


def _token_valid(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)


@web.middleware
async def admin_auth_middleware(request: web.Request, handler) -> web.StreamResponse:
    if request.method == "OPTIONS":
        return await handler(request)

    if not admin_auth_enforced():
        return await handler(request)

    path = request.path
    if not _path_requires_admin_auth(path):
        return await handler(request)

    secret = resolve_admin_secret()
    if not secret:
        logger.error("Admin auth enforced but no admin signing secret is configured")
        return web.json_response({"error": "admin auth misconfigured"}, status=503)

    token = _extract_bearer_token(request)
    if not _token_valid(token or "", secret):
        return web.json_response(
            {"error": "unauthorized", "message": "Valid admin bearer token required"},
            status=401,
        )

    return await handler(request)


def validate_production_startup() -> None:
    """Refuse startup on public admin bind without stable secrets and TLS posture."""
    admin_host = os.getenv("ADMIN_HOST", DEFAULT_ADMIN_HOST).strip()
    if is_loopback_host(admin_host):
        return

    if env_truthy("ALLOW_INSECURE_PUBLIC_BIND") is True:
        return

    problems: list[str] = []
    if not os.getenv("SIGNING_KEY", "").strip():
        problems.append("Set SIGNING_KEY in the environment (ephemeral keys break ledger continuity)")
    if not os.getenv("JWT_SECRET", "").strip():
        problems.append("Set JWT_SECRET in the environment (ephemeral keys invalidate node sessions)")
    if admin_auth_enforced():
        if not os.getenv("DISTRIBAI_ADMIN_SECRET", "").strip():
            problems.append("Set DISTRIBAI_ADMIN_SECRET for admin Bearer auth (do not reuse JWT_SECRET)")
    if os.getenv("GRPC_USE_TLS", "false").lower() != "true":
        problems.append(
            "Set GRPC_USE_TLS=true for public deployments, or set ALLOW_INSECURE_PUBLIC_BIND=1 "
            "only on private networks"
        )
    else:
        from services_python.grpc_tls import missing_server_tls_material

        tls_missing = missing_server_tls_material()
        if tls_missing:
            problems.append(
                "gRPC TLS is enabled but server certificate material is missing: "
                + ", ".join(tls_missing)
            )

    if problems:
        detail = "\n".join(f"  - {item}" for item in problems)
        logger.critical(
            "Refusing orchestrator startup: ADMIN_HOST=%s is not loopback.\n%s",
            admin_host,
            detail,
        )
        raise SystemExit(1)


def log_production_security_warnings() -> None:
    """Emit operator-visible warnings for insecure or ephemeral production defaults."""
    if not os.getenv("SIGNING_KEY", "").strip():
        logger.warning(
            "SIGNING_KEY is not set; this process generated an ephemeral signing key. "
            "Credit ledger continuity will not survive orchestrator restarts."
        )
    if not os.getenv("JWT_SECRET", "").strip():
        logger.warning(
            "JWT_SECRET is not set; this process generated an ephemeral JWT secret. "
            "Node sessions will be invalidated on orchestrator restart."
        )

    admin_host = os.getenv("ADMIN_HOST", DEFAULT_ADMIN_HOST).strip()
    if admin_auth_enforced():
        logger.info(
            "Admin API authentication is enforced (ADMIN_HOST=%s). "
            "Clients must send Authorization: Bearer <DISTRIBAI_ADMIN_SECRET>.",
            admin_host,
        )
    elif not is_loopback_host(admin_host):
        logger.warning(
            "ADMIN_HOST=%s but admin auth is not enforced. Set ADMIN_REQUIRE_AUTH=1 "
            "and DISTRIBAI_ADMIN_SECRET before exposing the admin API.",
            admin_host,
        )

    if os.getenv("GRPC_USE_TLS", "false").lower() != "true":
        if not is_loopback_host(admin_host):
            logger.warning(
                "gRPC is not using TLS while ADMIN_HOST=%s. Use GRPC_USE_TLS=true "
                "or restrict traffic to a private network.",
                admin_host,
            )

    if not is_loopback_host(admin_host):
        from services_python.cors_policy import cors_is_permissive

        if cors_is_permissive():
            logger.warning(
                "CORS_ALLOWED_ORIGINS is permissive while ADMIN_HOST=%s. "
                "Set explicit UI origins in production.",
                admin_host,
            )
