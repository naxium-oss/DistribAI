"""TLS / mTLS helpers shared by orchestrator listen and worker dial paths."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from services_python.network_policy import is_loopback_host

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CERT_PATH = _REPO_ROOT / "runtime" / "secrets" / "tls" / "server.crt"
_DEFAULT_KEY_PATH = _REPO_ROOT / "runtime" / "secrets" / "tls" / "server.key"


def _is_production() -> bool:
    return os.getenv("DISTRIBAI_ENV", "development").lower() == "production"


def grpc_tls_enabled() -> bool:
    return os.getenv("GRPC_USE_TLS", "true").lower() == "true"


def server_cert_path() -> Path:
    return Path(os.getenv("GRPC_TLS_CERT", str(_DEFAULT_CERT_PATH)))


def server_key_path() -> Path:
    return Path(os.getenv("GRPC_TLS_KEY", str(_DEFAULT_KEY_PATH)))


def grpc_bind_host() -> str:
    """Interface the orchestrator binds for gRPC (default all addresses)."""
    return os.getenv("GRPC_BIND_HOST", "0.0.0.0").strip()


def client_ca_path() -> Path | None:
    raw = os.getenv("GRPC_TLS_CA", "").strip()
    return Path(raw) if raw else None


def mtls_ca_path() -> Path | None:
    raw = os.getenv("GRPC_MTLS_CA", "").strip()
    if raw:
        return Path(raw)
    if grpc_require_client_cert():
        return client_ca_path()
    return None


def worker_client_cert_path() -> Path | None:
    raw = os.getenv("GRPC_TLS_CLIENT_CERT", "").strip()
    return Path(raw) if raw else None


def worker_client_key_path() -> Path | None:
    raw = os.getenv("GRPC_TLS_CLIENT_KEY", "").strip()
    return Path(raw) if raw else None


def grpc_require_client_cert() -> bool:
    """If set with TLS on, require presenting a client cert (mTLS)."""
    return os.getenv("GRPC_TLS_REQUIRE_CLIENT_CERT", "false").lower() == "true"


def missing_server_tls_material() -> list[str]:
    """List missing server TLS paths (empty when TLS is disabled)."""
    if not grpc_tls_enabled():
        return []
    missing: list[str] = []
    if not server_cert_path().is_file():
        missing.append(f"GRPC_TLS_CERT ({server_cert_path()})")
    if not server_key_path().is_file():
        missing.append(f"GRPC_TLS_KEY ({server_key_path()})")
    if grpc_require_client_cert():
        ca = client_ca_path()
        if ca is None or not ca.is_file():
            missing.append(f"GRPC_TLS_CA ({ca or 'unset'}) for mTLS")
    else:
        ca = mtls_ca_path()
        if ca is not None and not ca.is_file():
            missing.append(f"GRPC_MTLS_CA ({ca})")
    return missing


def _ensure_server_tls_material() -> None:
    cert_path = server_cert_path()
    key_path = server_key_path()
    if cert_path.is_file() and key_path.is_file():
        return
    if _is_production():
        raise RuntimeError(
            f"TLS enabled but cert ({cert_path}) or key ({key_path}) missing. "
            "Run: python scripts/dev/gen_tls_certs.py --hostname <your-host> --ca"
        )
    logger.warning(
        "TLS certs missing at %s / %s; auto-generating self-signed for dev.",
        cert_path,
        key_path,
    )
    from scripts.dev.gen_tls_certs import auto_generate_dev_cert

    auto_generate_dev_cert(cert_path, key_path)


def server_ssl_credentials():
    """Construct server SSL credentials (optionally with client-auth CA)."""
    import grpc

    _ensure_server_tls_material()
    private_key = server_key_path().read_bytes()
    cert_chain = server_cert_path().read_bytes()
    ca = mtls_ca_path()
    if ca is not None:
        if not ca.is_file():
            raise FileNotFoundError(f"mTLS CA missing at {ca}")
        return grpc.ssl_server_credentials(
            [(private_key, cert_chain)],
            root_certificates=ca.read_bytes(),
            require_client_auth=True,
        )
    return grpc.ssl_server_credentials([(private_key, cert_chain)])


def configure_grpc_server(server, grpc_port: str, grpc_listen: str | None = None) -> bool:
    """Attach a secure (default) or plaintext (dev-only) port to the gRPC server.

    Returns True for TLS bind, False when plaintext was allowed in development.
    """
    listen = grpc_listen if grpc_listen is not None else grpc_bind_host()
    bind_target = f"{listen}:{grpc_port}"

    if not grpc_tls_enabled():
        if _is_production():
            raise RuntimeError(
                "GRPC_USE_TLS=false rejected in production. "
                "Set DISTRIBAI_ENV=development to allow plaintext (dev only), "
                "or generate certs with `python scripts/dev/gen_tls_certs.py`."
            )
        logger.warning("gRPC plaintext bound (TLS disabled by GRPC_USE_TLS=false; dev mode).")
        server.add_insecure_port(bind_target)
        return False

    _ensure_server_tls_material()
    creds = server_ssl_credentials()
    ca = mtls_ca_path()
    if ca is not None:
        logger.info("gRPC mTLS enabled (client cert required, CA=%s).", ca)
    else:
        logger.info("gRPC TLS enabled (server cert=%s).", server_cert_path())

    server.add_secure_port(bind_target, creds)
    return True


def orchestrator_grpc_host_public() -> bool:
    """Whether the gRPC bind host is reachable beyond loopback."""
    return not is_loopback_host(grpc_bind_host())


def parse_grpc_target_host(target: str) -> str:
    """Parse the host portion of a gRPC target string (host:port or URI-ish)."""
    cleaned = target.replace("http://", "").replace("https://", "").replace("ws://", "")
    if cleaned.startswith("["):
        end = cleaned.find("]")
        if end != -1:
            return cleaned[1:end]
    host = cleaned.split(":", 1)[0]
    return host.strip().lower()


def grpc_target_is_public(target: str) -> bool:
    host = parse_grpc_target_host(target)
    if not host:
        return False
    return not is_loopback_host(host)
