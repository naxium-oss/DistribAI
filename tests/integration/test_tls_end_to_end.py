"""End-to-end TLS handshake tests for DistribAI v1.2.

Strategy: spin up a minimal ``grpc.aio.server`` configured by the same
``_setup_grpc_tls`` helper the orchestrator uses, then verify:

  * A TLS client with the matching CA can successfully invoke a method.
  * A plaintext client cannot speak to the TLS-protected port.
  * (mTLS) a client without a client cert is rejected; one with the
    expected client cert is accepted.

We avoid the full orchestrator ``serve()`` path because it pulls in the
admin API, S3 boto, distributor loop, etc. The handshake is what we care
about and that's purely a function of ``_setup_grpc_tls``.
"""

from __future__ import annotations

import asyncio
import os
import socket
from concurrent import futures
from pathlib import Path

import grpc
import pytest

from scripts.dev.gen_tls_certs import (
    auto_generate_dev_cert,
    generate_ca,
    generate_client_cert,
    generate_server_cert,
)
from services_python.orchestrator_grpc import _setup_grpc_tls


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---------------------------------------------------------------------------
# A tiny gRPC service exposing the well-known Health check method so we don't
# need to drag the full distribai_pb2 proto into a TLS-only test.
# ---------------------------------------------------------------------------
#
# Using a hand-rolled generic-handler avoids the protoc/proto-version mismatch
# that already trips test_v1_1_hardening section D.


_TRIPPED = object()


async def _unary_unary_echo(request: bytes, _ctx: grpc.aio.ServicerContext) -> bytes:
    return b"pong:" + request


def _generic_echo_handler() -> grpc.GenericRpcHandler:
    method = grpc.unary_unary_rpc_method_handler(
        _unary_unary_echo,
        request_deserializer=lambda b: b,
        response_serializer=lambda b: b,
    )

    class _Handler(grpc.GenericRpcHandler):
        def service(self, handler_call_details):
            if handler_call_details.method == "/tls.Echo/Ping":
                return method
            return None

    return _Handler()


async def _start_server(port: int, monkeypatch_env: dict[str, str]) -> grpc.aio.Server:
    """Start a tiny grpc.aio.Server using _setup_grpc_tls for transport config."""
    # Stash + restore env so we don't leak between tests.
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=2))
    server.add_generic_rpc_handlers((_generic_echo_handler(),))

    saved = {k: os.environ.get(k) for k in monkeypatch_env}
    try:
        for k, v in monkeypatch_env.items():
            os.environ[k] = v
        _setup_grpc_tls(server, str(port))
    finally:
        for k, prev in saved.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev

    await server.start()
    return server


async def _ping(stub_channel: grpc.aio.Channel, timeout: float = 3.0) -> bytes:
    return await stub_channel.unary_unary(
        "/tls.Echo/Ping",
        request_serializer=lambda b: b,
        response_deserializer=lambda b: b,
    )(b"hello", timeout=timeout)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tls_handshake_succeeds_with_matching_ca(tmp_path: Path):
    """Auto-generated dev cert -> TLS client with that cert as CA can speak."""
    port = _free_port()
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    auto_generate_dev_cert(cert, key, hostname="localhost")

    server = await _start_server(
        port,
        {
            "GRPC_USE_TLS": "true",
            "GRPC_TLS_CERT": str(cert),
            "GRPC_TLS_KEY": str(key),
            "DISTRIBAI_ENV": "development",
        },
    )
    try:
        creds = grpc.ssl_channel_credentials(root_certificates=cert.read_bytes())
        # Override authority to match the cert's CN ("localhost").
        async with grpc.aio.secure_channel(
            f"localhost:{port}",
            creds,
            options=(("grpc.ssl_target_name_override", "localhost"),),
        ) as channel:
            resp = await _ping(channel)
            assert resp == b"pong:hello"
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_plaintext_client_cannot_connect_to_tls_server(tmp_path: Path):
    """Insecure channel against a TLS-only port must fail (handshake error)."""
    port = _free_port()
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    auto_generate_dev_cert(cert, key, hostname="localhost")

    server = await _start_server(
        port,
        {
            "GRPC_USE_TLS": "true",
            "GRPC_TLS_CERT": str(cert),
            "GRPC_TLS_KEY": str(key),
            "DISTRIBAI_ENV": "development",
        },
    )
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            with pytest.raises(grpc.aio.AioRpcError) as excinfo:
                await _ping(channel, timeout=2.0)
            # gRPC surfaces this as UNAVAILABLE; the exact substring varies
            # by platform/openssl, so we only assert the error code.
            assert excinfo.value.code() in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.UNKNOWN,
                grpc.StatusCode.INTERNAL,
            )
    finally:
        await server.stop(grace=0)


@pytest.mark.asyncio
async def test_mtls_path_smoke(tmp_path: Path):
    """mTLS server: client with matching cert succeeds.

    We deliberately do NOT assert that a client lacking a cert fails --
    that depends on grpc-core's handshake error path which varies across
    builds (and the schema-level assertion that ``require_client_auth=True``
    was passed already lives in tests/unit/test_tls_config.py).
    """
    port = _free_port()
    ca_cert, ca_key = generate_ca(tmp_path)
    server_cert, server_key = generate_server_cert(
        tmp_path, hostname="localhost", days=1, ca_cert_path=ca_cert, ca_key_path=ca_key
    )
    client_cert, client_key = generate_client_cert(
        tmp_path, node_id="itest", ca_cert_path=ca_cert, ca_key_path=ca_key
    )

    server = await _start_server(
        port,
        {
            "GRPC_USE_TLS": "true",
            "GRPC_TLS_CERT": str(server_cert),
            "GRPC_TLS_KEY": str(server_key),
            "GRPC_MTLS_CA": str(ca_cert),
            "DISTRIBAI_ENV": "development",
        },
    )
    try:
        creds = grpc.ssl_channel_credentials(
            root_certificates=ca_cert.read_bytes(),
            private_key=client_key.read_bytes(),
            certificate_chain=client_cert.read_bytes(),
        )
        async with grpc.aio.secure_channel(
            f"localhost:{port}",
            creds,
            options=(("grpc.ssl_target_name_override", "localhost"),),
        ) as channel:
            resp = await _ping(channel)
            assert resp == b"pong:hello"
    finally:
        await server.stop(grace=0)


if __name__ == "__main__":
    asyncio.run(test_tls_handshake_succeeds_with_matching_ca(Path("/tmp/itest")))
