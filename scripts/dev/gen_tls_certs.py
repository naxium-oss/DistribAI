"""Generate self-signed TLS certs (and optionally a CA + mTLS client certs) for DistribAI.

Usage:
    python scripts/dev/gen_tls_certs.py
        # Self-signed dev cert, RSA 4096, 365 days, hostname=localhost.
    python scripts/dev/gen_tls_certs.py --hostname grid.acme.io --days 365
    python scripts/dev/gen_tls_certs.py --ca
        # Also issue a root CA. server.crt is then signed by ca.crt.
    python scripts/dev/gen_tls_certs.py --mtls --node-id worker-42
        # Requires --ca to also be present (or an existing ca.crt + ca.key
        # in the output directory). Emits worker-<id>.crt + worker-<id>.key
        # for use as a gRPC client certificate.

Outputs under <repo>/runtime/secrets/tls/ (override with --out-dir):
    ca.crt           # root CA cert  (when --ca)
    ca.key           # root CA key   (when --ca)             [mode 0600]
    server.crt       # orchestrator server cert
    server.key       # orchestrator server private key       [mode 0600]
    worker-<id>.crt  # per-node client cert (when --mtls)
    worker-<id>.key  # per-node client key (when --mtls)     [mode 0600]

This module is also imported by services_python/orchestrator_grpc.py:
``auto_generate_dev_cert(cert_path, key_path)`` is invoked when the
orchestrator boots with TLS enabled, dev mode, and no certs on disk.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import ipaddress
import logging
import os
import stat
import sys
from collections.abc import Iterable
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Resolve <repo>/runtime/secrets/tls relative to this file so the helper is
# importable from anywhere on PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = _REPO_ROOT / "runtime" / "secrets" / "tls"
DEFAULT_HOSTNAME = "localhost"
DEFAULT_DAYS = 365
DEFAULT_RSA_BITS = 4096
DEFAULT_ALGO = "rsa"  # or "ecdsa"


# ---------------------------------------------------------------------------
# Key + cert primitives
# ---------------------------------------------------------------------------


PrivateKey = rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey  # type: ignore[valid-type]


def _generate_private_key(algo: str = DEFAULT_ALGO) -> PrivateKey:
    """Generate an RSA-4096 or ECDSA-P256 private key."""
    if algo == "rsa":
        return rsa.generate_private_key(public_exponent=65537, key_size=DEFAULT_RSA_BITS)
    if algo == "ecdsa":
        return ec.generate_private_key(ec.SECP256R1())
    raise ValueError(f"unsupported algo: {algo!r} (expected 'rsa' or 'ecdsa')")


def _build_sans(hostname: str) -> x509.SubjectAlternativeName:
    """Build SAN extension with the hostname + localhost + 127.0.0.1.

    The IP/DNS distinction matters for gRPC peer verification, so we
    duck-type-check whether ``hostname`` parses as an IP address.
    """
    sans: list[x509.GeneralName] = []
    try:
        ip = ipaddress.ip_address(hostname)
        sans.append(x509.IPAddress(ip))
    except ValueError:
        sans.append(x509.DNSName(hostname))

    # Always include localhost + 127.0.0.1 so dev clients can connect.
    if hostname != "localhost":
        sans.append(x509.DNSName("localhost"))
    if hostname != "127.0.0.1":
        sans.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))

    return x509.SubjectAlternativeName(sans)


def _make_name(common_name: str, organization: str = "DistribAI") -> x509.Name:
    return x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ]
    )


def _build_ca_cert(
    ca_key: PrivateKey, common_name: str = "DistribAI Local CA", days: int = 3650
) -> x509.Certificate:
    """Build a self-signed root CA certificate."""
    name = _make_name(common_name, organization="DistribAI CA")
    now = _dt.datetime.now(_dt.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=1))
        .not_valid_after(now + _dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
    )
    return builder.sign(ca_key, hashes.SHA256())


def _build_server_cert(
    server_key: PrivateKey,
    hostname: str,
    days: int,
    issuer_cert: x509.Certificate | None,
    issuer_key: PrivateKey | None,
) -> x509.Certificate:
    """Build a server cert. If ``issuer_*`` is None, the cert is self-signed."""
    subject = _make_name(hostname)
    if issuer_cert is None or issuer_key is None:
        issuer = subject
        signing_key = server_key
    else:
        issuer = issuer_cert.subject
        signing_key = issuer_key

    now = _dt.datetime.now(_dt.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=1))
        .not_valid_after(now + _dt.timedelta(days=days))
        .add_extension(_build_sans(hostname), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
    )
    return builder.sign(signing_key, hashes.SHA256())


def _build_client_cert(
    client_key: PrivateKey,
    node_id: str,
    days: int,
    issuer_cert: x509.Certificate,
    issuer_key: PrivateKey,
) -> x509.Certificate:
    """Build a client cert (mTLS) signed by the given CA."""
    subject = _make_name(node_id, organization="DistribAI Worker")
    now = _dt.datetime.now(_dt.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_cert.subject)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - _dt.timedelta(minutes=1))
        .not_valid_after(now + _dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
    )
    return builder.sign(issuer_key, hashes.SHA256())


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------


def _write_key(key: PrivateKey, path: Path) -> None:
    """Write a PEM-encoded unencrypted private key with mode 0600 on POSIX."""
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and atomically replace so concurrent readers
    # never see a half-written key.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(pem)
    try:
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError:
        pass  # Windows: chmod is a no-op.
    os.replace(tmp, path)


def _write_cert(cert: x509.Certificate, path: Path) -> None:
    pem = cert.public_bytes(serialization.Encoding.PEM)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(pem)
    os.replace(tmp, path)


def _load_key(path: Path) -> PrivateKey:
    return serialization.load_pem_private_key(path.read_bytes(), password=None)  # type: ignore[return-value]


def _load_cert(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


# ---------------------------------------------------------------------------
# Public entry points (importable + CLI)
# ---------------------------------------------------------------------------


def generate_ca(out_dir: Path, algo: str = DEFAULT_ALGO, days: int = 3650) -> tuple[Path, Path]:
    """Generate (or reuse) a root CA. Returns (ca_cert_path, ca_key_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ca_cert_path = out_dir / "ca.crt"
    ca_key_path = out_dir / "ca.key"
    if ca_cert_path.exists() and ca_key_path.exists():
        logger.info("Reusing existing CA at %s", ca_cert_path)
        return ca_cert_path, ca_key_path

    ca_key = _generate_private_key(algo)
    ca_cert = _build_ca_cert(ca_key, days=days)
    _write_cert(ca_cert, ca_cert_path)
    _write_key(ca_key, ca_key_path)
    logger.info("Wrote CA: %s (key: %s)", ca_cert_path, ca_key_path)
    return ca_cert_path, ca_key_path


def generate_server_cert(
    out_dir: Path,
    hostname: str = DEFAULT_HOSTNAME,
    days: int = DEFAULT_DAYS,
    algo: str = DEFAULT_ALGO,
    ca_cert_path: Path | None = None,
    ca_key_path: Path | None = None,
) -> tuple[Path, Path]:
    """Generate the orchestrator server cert. Returns (cert_path, key_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cert_path = out_dir / "server.crt"
    key_path = out_dir / "server.key"

    server_key = _generate_private_key(algo)
    if ca_cert_path and ca_key_path:
        ca_cert = _load_cert(ca_cert_path)
        ca_key = _load_key(ca_key_path)
        server_cert = _build_server_cert(server_key, hostname, days, ca_cert, ca_key)
    else:
        server_cert = _build_server_cert(server_key, hostname, days, None, None)
    _write_cert(server_cert, cert_path)
    _write_key(server_key, key_path)
    logger.info("Wrote server cert: %s (key: %s)", cert_path, key_path)
    return cert_path, key_path


def generate_client_cert(
    out_dir: Path,
    node_id: str,
    ca_cert_path: Path,
    ca_key_path: Path,
    days: int = DEFAULT_DAYS,
    algo: str = DEFAULT_ALGO,
) -> tuple[Path, Path]:
    """Generate an mTLS client cert signed by the supplied CA."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cert_path = out_dir / f"worker-{node_id}.crt"
    key_path = out_dir / f"worker-{node_id}.key"

    client_key = _generate_private_key(algo)
    ca_cert = _load_cert(ca_cert_path)
    ca_key = _load_key(ca_key_path)
    client_cert = _build_client_cert(client_key, node_id, days, ca_cert, ca_key)
    _write_cert(client_cert, cert_path)
    _write_key(client_key, key_path)
    logger.info("Wrote client cert: %s (key: %s)", cert_path, key_path)
    return cert_path, key_path


def auto_generate_dev_cert(cert_path: Path, key_path: Path, hostname: str = DEFAULT_HOSTNAME) -> None:
    """Generate a self-signed dev cert at the given paths if absent.

    Imported by services_python/orchestrator_grpc.py to satisfy the
    fail-soft-on-dev / fail-closed-on-prod boot rule. Always RSA 4096 and
    365 days; if you want ECDSA or a different lifetime, run the CLI
    yourself before booting.
    """
    cert_path = Path(cert_path)
    key_path = Path(key_path)
    cert_path.parent.mkdir(parents=True, exist_ok=True)

    server_key = _generate_private_key("rsa")
    server_cert = _build_server_cert(server_key, hostname, DEFAULT_DAYS, None, None)
    _write_cert(server_cert, cert_path)
    _write_key(server_key, key_path)
    logger.warning(
        "Auto-generated self-signed dev cert at %s (key: %s). "
        "DO NOT use these for production -- run scripts/dev/gen_tls_certs.py "
        "with --hostname and --ca for a real deployment.",
        cert_path,
        key_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gen_tls_certs",
        description="Generate TLS certs for DistribAI (self-signed + optional CA + mTLS).",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory (default: runtime/secrets/tls/)")
    p.add_argument("--hostname", default=DEFAULT_HOSTNAME, help="Server hostname or IP for SAN (default: localhost)")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Server cert validity in days (default: 365)")
    p.add_argument("--algo", choices=("rsa", "ecdsa"), default=DEFAULT_ALGO, help="Key algorithm (default: rsa)")
    p.add_argument("--ca", action="store_true", help="Also issue a root CA; sign the server cert with it.")
    p.add_argument("--mtls", action="store_true", help="Issue a client cert for mTLS (requires --node-id + a CA).")
    p.add_argument("--node-id", default=None, help="Worker node id for the mTLS client cert.")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ca_paths: tuple[Path, Path] | None = None
    if args.ca:
        ca_paths = generate_ca(out_dir, algo=args.algo)
    elif args.mtls:
        # mTLS requires a CA; auto-create one if absent so single-command
        # `--mtls --node-id alice` still works.
        ca_cert = out_dir / "ca.crt"
        ca_key = out_dir / "ca.key"
        if not (ca_cert.exists() and ca_key.exists()):
            logger.info("--mtls requested without --ca; auto-creating local CA.")
            ca_paths = generate_ca(out_dir, algo=args.algo)
        else:
            ca_paths = (ca_cert, ca_key)

    server_paths = generate_server_cert(
        out_dir,
        hostname=args.hostname,
        days=args.days,
        algo=args.algo,
        ca_cert_path=ca_paths[0] if ca_paths else None,
        ca_key_path=ca_paths[1] if ca_paths else None,
    )

    if args.mtls:
        if not args.node_id:
            logger.error("--mtls requires --node-id <id>.")
            return 2
        assert ca_paths is not None  # guarded above
        generate_client_cert(
            out_dir,
            node_id=args.node_id,
            ca_cert_path=ca_paths[0],
            ca_key_path=ca_paths[1],
            days=args.days,
            algo=args.algo,
        )

    print(f"Done. Server cert: {server_paths[0]}")
    if ca_paths:
        print(f"      Root CA:    {ca_paths[0]}  (distribute to workers as GRPC_TLS_CA)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
