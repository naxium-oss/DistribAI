"""Unit tests for services_python.orchestrator_grpc TLS config helpers.

Covers the matrix of (GRPC_USE_TLS x DISTRIBAI_ENV x certs-present x mTLS)
behaviour for ``_setup_grpc_tls`` and ``_setup_admin_tls``.
"""

from __future__ import annotations

import ssl
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.dev.gen_tls_certs import auto_generate_dev_cert

# Defer importing the orchestrator module until paths are set so we don't
# trip over slow side-effects in collection.
from services_python.orchestrator_grpc import (
    _DEFAULT_CERT_PATH,
    _DEFAULT_KEY_PATH,
    _setup_admin_tls,
    _setup_grpc_tls,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def _fake_server() -> MagicMock:
    """A MagicMock standing in for ``grpc.aio.server()``.

    We don't need a real server -- we only assert which ``add_*_port``
    method is called and with what credentials.
    """
    return MagicMock()


@pytest.fixture
def _isolate_env(monkeypatch: pytest.MonkeyPatch):
    """Clear any TLS-related env vars so each test starts from a clean slate."""
    for k in [
        "GRPC_USE_TLS",
        "GRPC_TLS_CERT",
        "GRPC_TLS_KEY",
        "GRPC_MTLS_CA",
        "DISTRIBAI_ENV",
        "ADMIN_USE_TLS",
        "ADMIN_TLS_CERT",
        "ADMIN_TLS_KEY",
    ]:
        monkeypatch.delenv(k, raising=False)


# ---------------------------------------------------------------------------
# GRPC_USE_TLS=false branch
# ---------------------------------------------------------------------------


def test_grpc_use_tls_false_in_production_raises(_fake_server, _isolate_env, monkeypatch):
    monkeypatch.setenv("GRPC_USE_TLS", "false")
    monkeypatch.setenv("DISTRIBAI_ENV", "production")
    with pytest.raises(RuntimeError, match="GRPC_USE_TLS=false rejected in production"):
        _setup_grpc_tls(_fake_server, "50051")
    _fake_server.add_insecure_port.assert_not_called()
    _fake_server.add_secure_port.assert_not_called()


def test_grpc_use_tls_false_in_dev_falls_through_to_insecure(_fake_server, _isolate_env, monkeypatch):
    monkeypatch.setenv("GRPC_USE_TLS", "false")
    # DISTRIBAI_ENV unset -> defaults to development.
    result = _setup_grpc_tls(_fake_server, "50051")
    assert result is False
    _fake_server.add_insecure_port.assert_called_once_with("[::]:50051")
    _fake_server.add_secure_port.assert_not_called()


# ---------------------------------------------------------------------------
# GRPC_USE_TLS=true + missing certs
# ---------------------------------------------------------------------------


def test_missing_certs_in_production_raises(_fake_server, _isolate_env, monkeypatch, tmp_path):
    monkeypatch.setenv("DISTRIBAI_ENV", "production")
    monkeypatch.setenv("GRPC_TLS_CERT", str(tmp_path / "nope.crt"))
    monkeypatch.setenv("GRPC_TLS_KEY", str(tmp_path / "nope.key"))
    with pytest.raises(RuntimeError, match="cert .* or key .* missing"):
        _setup_grpc_tls(_fake_server, "50051")


def test_missing_certs_in_dev_auto_generates(_fake_server, _isolate_env, monkeypatch, tmp_path):
    cert = tmp_path / "auto.crt"
    key = tmp_path / "auto.key"
    monkeypatch.setenv("GRPC_TLS_CERT", str(cert))
    monkeypatch.setenv("GRPC_TLS_KEY", str(key))
    # DISTRIBAI_ENV unset -> dev.

    result = _setup_grpc_tls(_fake_server, "50051")
    assert result is True
    assert cert.exists() and key.exists()
    _fake_server.add_secure_port.assert_called_once()
    args, _kwargs = _fake_server.add_secure_port.call_args
    assert args[0] == "[::]:50051"
    # The credentials object is a grpc.ServerCredentials -- opaque, just
    # assert it's not the insecure path.
    _fake_server.add_insecure_port.assert_not_called()


# ---------------------------------------------------------------------------
# GRPC_USE_TLS=true + valid certs (happy path) + mTLS branch
# ---------------------------------------------------------------------------


def test_valid_certs_configure_secure_port(_fake_server, _isolate_env, monkeypatch, tmp_path):
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    auto_generate_dev_cert(cert, key)
    monkeypatch.setenv("GRPC_TLS_CERT", str(cert))
    monkeypatch.setenv("GRPC_TLS_KEY", str(key))

    result = _setup_grpc_tls(_fake_server, "50052")
    assert result is True
    _fake_server.add_secure_port.assert_called_once()
    _fake_server.add_insecure_port.assert_not_called()


def test_mtls_path_uses_require_client_auth(_fake_server, _isolate_env, monkeypatch, tmp_path):
    from scripts.dev.gen_tls_certs import generate_ca, generate_server_cert

    ca_cert, ca_key = generate_ca(tmp_path)
    server_cert, server_key = generate_server_cert(
        tmp_path, hostname="mtls.test", days=1, ca_cert_path=ca_cert, ca_key_path=ca_key
    )
    monkeypatch.setenv("GRPC_TLS_CERT", str(server_cert))
    monkeypatch.setenv("GRPC_TLS_KEY", str(server_key))
    monkeypatch.setenv("GRPC_MTLS_CA", str(ca_cert))

    # Patch grpc.ssl_server_credentials so we can assert require_client_auth=True
    # without touching the real C-extension constructor.
    with patch("grpc.ssl_server_credentials") as mocked:
        sentinel = object()
        mocked.return_value = sentinel
        _setup_grpc_tls(_fake_server, "50053")
        assert mocked.called
        _, kwargs = mocked.call_args
        assert kwargs["require_client_auth"] is True
        assert kwargs["root_certificates"] == ca_cert.read_bytes()
    _fake_server.add_secure_port.assert_called_once_with("[::]:50053", sentinel)


# ---------------------------------------------------------------------------
# _setup_admin_tls
# ---------------------------------------------------------------------------


def test_admin_tls_disabled_for_loopback_by_default(_isolate_env):
    assert _setup_admin_tls("127.0.0.1") is None
    assert _setup_admin_tls("localhost") is None
    assert _setup_admin_tls("::1") is None


def test_admin_tls_auto_enables_for_non_loopback(_isolate_env, monkeypatch, tmp_path):
    cert = tmp_path / "admin.crt"
    key = tmp_path / "admin.key"
    auto_generate_dev_cert(cert, key)
    monkeypatch.setenv("ADMIN_TLS_CERT", str(cert))
    monkeypatch.setenv("ADMIN_TLS_KEY", str(key))
    ctx = _setup_admin_tls("0.0.0.0")
    assert isinstance(ctx, ssl.SSLContext)


def test_admin_tls_explicit_false_on_non_loopback_in_prod_raises(_isolate_env, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ENV", "production")
    monkeypatch.setenv("ADMIN_USE_TLS", "false")
    with pytest.raises(RuntimeError, match="Admin API bound to non-loopback"):
        _setup_admin_tls("0.0.0.0")


def test_admin_tls_explicit_false_on_non_loopback_in_dev_ok(_isolate_env, monkeypatch):
    monkeypatch.setenv("ADMIN_USE_TLS", "false")
    # dev mode
    assert _setup_admin_tls("0.0.0.0") is None


def test_admin_tls_missing_certs_dev_autogens(_isolate_env, monkeypatch, tmp_path):
    cert = tmp_path / "missing.crt"
    key = tmp_path / "missing.key"
    monkeypatch.setenv("ADMIN_TLS_CERT", str(cert))
    monkeypatch.setenv("ADMIN_TLS_KEY", str(key))
    ctx = _setup_admin_tls("0.0.0.0")
    assert isinstance(ctx, ssl.SSLContext)
    assert cert.exists() and key.exists()


def test_admin_tls_missing_certs_prod_raises(_isolate_env, monkeypatch, tmp_path):
    monkeypatch.setenv("DISTRIBAI_ENV", "production")
    monkeypatch.setenv("ADMIN_TLS_CERT", str(tmp_path / "x.crt"))
    monkeypatch.setenv("ADMIN_TLS_KEY", str(tmp_path / "x.key"))
    with pytest.raises(RuntimeError, match="Admin TLS enabled but cert"):
        _setup_admin_tls("0.0.0.0")


# ---------------------------------------------------------------------------
# Defaults sanity check
# ---------------------------------------------------------------------------


def test_default_cert_paths_under_runtime_secrets_tls():
    """The defaults must line up with scripts/gen_tls_certs.py output."""
    assert _DEFAULT_CERT_PATH.name == "server.crt"
    assert _DEFAULT_KEY_PATH.name == "server.key"
    assert _DEFAULT_CERT_PATH.parent.name == "tls"
    assert _DEFAULT_CERT_PATH.parent.parent.name == "secrets"
