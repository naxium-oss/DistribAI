"""Tests for gRPC TLS material helpers."""

from __future__ import annotations

from services_python.grpc_tls import (
    grpc_bind_host,
    grpc_require_client_cert,
    grpc_target_is_public,
    grpc_tls_enabled,
    missing_server_tls_material,
    orchestrator_grpc_host_public,
    parse_grpc_target_host,
)


def test_parse_grpc_target_host():
    assert parse_grpc_target_host("localhost:50051") == "localhost"
    assert parse_grpc_target_host("[::1]:50051") == "::1"


def test_grpc_target_is_public():
    assert grpc_target_is_public("203.0.113.10:50051") is True
    assert grpc_target_is_public("127.0.0.1:50051") is False
    assert grpc_target_is_public("") is False
    assert grpc_target_is_public(":50051") is False


def test_grpc_bind_host_and_public_bind(monkeypatch):
    monkeypatch.setenv("GRPC_BIND_HOST", "127.0.0.1")
    assert grpc_bind_host() == "127.0.0.1"
    assert orchestrator_grpc_host_public() is False
    monkeypatch.setenv("GRPC_BIND_HOST", "0.0.0.0")
    assert orchestrator_grpc_host_public() is True


def test_missing_server_tls_material_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("GRPC_USE_TLS", "true")
    monkeypatch.setenv("GRPC_TLS_CERT", str(tmp_path / "missing.crt"))
    monkeypatch.setenv("GRPC_TLS_KEY", str(tmp_path / "missing.key"))
    assert grpc_tls_enabled() is True
    missing = missing_server_tls_material()
    assert len(missing) == 2


def test_missing_server_tls_material_none_when_disabled(monkeypatch):
    monkeypatch.setenv("GRPC_USE_TLS", "false")
    assert missing_server_tls_material() == []


def test_missing_server_tls_material_empty_when_files_exist(monkeypatch, tmp_path):
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")
    monkeypatch.setenv("GRPC_USE_TLS", "true")
    monkeypatch.setenv("GRPC_TLS_CERT", str(cert))
    monkeypatch.setenv("GRPC_TLS_KEY", str(key))
    assert missing_server_tls_material() == []


def test_mtls_requires_client_ca_when_enabled(monkeypatch, tmp_path):
    cert = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    cert.write_text("cert", encoding="utf-8")
    key.write_text("key", encoding="utf-8")
    monkeypatch.setenv("GRPC_USE_TLS", "true")
    monkeypatch.setenv("GRPC_TLS_REQUIRE_CLIENT_CERT", "true")
    monkeypatch.setenv("GRPC_TLS_CERT", str(cert))
    monkeypatch.setenv("GRPC_TLS_KEY", str(key))
    assert grpc_require_client_cert() is True
    missing = missing_server_tls_material()
    assert any("GRPC_TLS_CA" in item for item in missing)
