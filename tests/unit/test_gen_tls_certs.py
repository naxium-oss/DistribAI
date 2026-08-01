"""Unit tests for scripts/gen_tls_certs.py.

Covers:
  * --hostname / --days produces a SAN-correct cert with the requested lifetime.
  * --ca produces a self-signed CA that signs the server cert.
  * --mtls --node-id produces a client cert with the node-id CN.
  * Key files are written with mode 0600 on POSIX.
  * auto_generate_dev_cert() (the orchestrator boot hook) writes both files.
"""

from __future__ import annotations

import datetime as _dt
import os
import stat
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

REPO_ROOT = Path(__file__).resolve().parents[2]

# Importable helpers under test (also exercised through the CLI further down).
from scripts.dev.gen_tls_certs import (  # noqa: E402, I001
    auto_generate_dev_cert,
    generate_ca,
    generate_client_cert,
    generate_server_cert,
    main as cli_main,
)


def _load_cert(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _san_values(cert: x509.Certificate) -> list[str]:
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    out: list[str] = []
    for name in san:
        if isinstance(name, x509.DNSName):
            out.append(f"DNS:{name.value}")
        elif isinstance(name, x509.IPAddress):
            out.append(f"IP:{name.value}")
    return out


def _common_name(cert: x509.Certificate) -> str:
    return cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value


def _is_posix() -> bool:
    return os.name == "posix"


# ---------------------------------------------------------------------------
# generate_server_cert (the workhorse) -- direct API
# ---------------------------------------------------------------------------


def test_server_cert_has_hostname_san_and_validity(tmp_path: Path):
    cert_path, key_path = generate_server_cert(tmp_path, hostname="foo.example.com", days=1)
    assert cert_path.exists() and key_path.exists()

    cert = _load_cert(cert_path)
    sans = _san_values(cert)
    assert "DNS:foo.example.com" in sans
    assert "DNS:localhost" in sans
    assert "IP:127.0.0.1" in sans

    # Validity window roughly 1 day (+/- a small backdate).
    delta = cert.not_valid_after_utc - cert.not_valid_before_utc
    assert _dt.timedelta(hours=23) <= delta <= _dt.timedelta(days=1, hours=1)


def test_server_cert_ip_hostname_becomes_ip_san(tmp_path: Path):
    cert_path, _ = generate_server_cert(tmp_path, hostname="10.0.0.5", days=1)
    sans = _san_values(_load_cert(cert_path))
    assert "IP:10.0.0.5" in sans


def _assert_key_mode_0600(key_path: Path) -> None:
    """POSIX: key must be owner-only. Windows: chmod is best-effort."""
    assert key_path.exists()
    if not _is_posix():
        return
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_server_key_mode_0600(tmp_path: Path):
    _, key_path = generate_server_cert(tmp_path, hostname="x.test", days=1)
    _assert_key_mode_0600(key_path)


# ---------------------------------------------------------------------------
# generate_ca + signed server cert
# ---------------------------------------------------------------------------


def test_ca_signs_server_cert(tmp_path: Path):
    ca_cert_path, ca_key_path = generate_ca(tmp_path)
    server_cert_path, _ = generate_server_cert(
        tmp_path, hostname="server.test", days=30, ca_cert_path=ca_cert_path, ca_key_path=ca_key_path
    )
    ca_cert = _load_cert(ca_cert_path)
    server_cert = _load_cert(server_cert_path)

    # Issuer of server cert must equal subject of CA cert.
    assert server_cert.issuer == ca_cert.subject

    # CA must have basicConstraints CA=True.
    bc = ca_cert.extensions.get_extension_for_class(x509.BasicConstraints).value
    assert bc.ca is True

    # Cryptographic verification: server cert signature checks against CA pubkey.
    from cryptography.hazmat.primitives.asymmetric import padding

    ca_cert.public_key().verify(
        server_cert.signature,
        server_cert.tbs_certificate_bytes,
        padding.PKCS1v15(),
        server_cert.signature_hash_algorithm,
    )


def test_ca_key_mode_0600(tmp_path: Path):
    _, ca_key_path = generate_ca(tmp_path)
    _assert_key_mode_0600(ca_key_path)


# ---------------------------------------------------------------------------
# generate_client_cert (mTLS)
# ---------------------------------------------------------------------------


def test_client_cert_cn_and_eku(tmp_path: Path):
    ca_cert_path, ca_key_path = generate_ca(tmp_path)
    client_cert_path, client_key_path = generate_client_cert(
        tmp_path, node_id="alice", ca_cert_path=ca_cert_path, ca_key_path=ca_key_path
    )
    assert client_cert_path.name == "worker-alice.crt"
    assert client_key_path.name == "worker-alice.key"

    cert = _load_cert(client_cert_path)
    assert _common_name(cert) == "alice"

    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert ExtendedKeyUsageOID.CLIENT_AUTH in eku


def test_client_key_mode_0600(tmp_path: Path):
    ca_cert_path, ca_key_path = generate_ca(tmp_path)
    _, client_key_path = generate_client_cert(
        tmp_path, node_id="bob", ca_cert_path=ca_cert_path, ca_key_path=ca_key_path
    )
    _assert_key_mode_0600(client_key_path)


# ---------------------------------------------------------------------------
# auto_generate_dev_cert -- orchestrator boot hook
# ---------------------------------------------------------------------------


def test_auto_generate_dev_cert_writes_both(tmp_path: Path):
    cert = tmp_path / "nested" / "server.crt"
    key = tmp_path / "nested" / "server.key"
    assert not cert.exists()
    auto_generate_dev_cert(cert, key, hostname="dev.local")
    assert cert.exists() and key.exists()

    parsed = _load_cert(cert)
    sans = _san_values(parsed)
    assert "DNS:dev.local" in sans


def test_auto_generate_dev_cert_key_mode_0600(tmp_path: Path):
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    auto_generate_dev_cert(cert, key)
    _assert_key_mode_0600(key)


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_main_self_signed(tmp_path: Path, capsys):
    rc = cli_main(["--out-dir", str(tmp_path), "--hostname", "cli.test", "--days", "1"])
    assert rc == 0
    assert (tmp_path / "server.crt").exists()
    assert (tmp_path / "server.key").exists()
    assert not (tmp_path / "ca.crt").exists()


def test_cli_main_mtls_autocreates_ca(tmp_path: Path):
    rc = cli_main(["--out-dir", str(tmp_path), "--mtls", "--node-id", "carol", "--days", "1"])
    assert rc == 0
    assert (tmp_path / "ca.crt").exists()
    assert (tmp_path / "ca.key").exists()
    assert (tmp_path / "server.crt").exists()
    assert (tmp_path / "worker-carol.crt").exists()
    assert (tmp_path / "worker-carol.key").exists()


def test_cli_main_mtls_without_node_id_returns_2(tmp_path: Path):
    rc = cli_main(["--out-dir", str(tmp_path), "--mtls"])
    assert rc == 2


def test_cli_main_ecdsa(tmp_path: Path):
    rc = cli_main(["--out-dir", str(tmp_path), "--algo", "ecdsa", "--hostname", "ec.test", "--days", "1"])
    assert rc == 0
    key = serialization.load_pem_private_key((tmp_path / "server.key").read_bytes(), password=None)
    from cryptography.hazmat.primitives.asymmetric import ec

    assert isinstance(key, ec.EllipticCurvePrivateKey)
