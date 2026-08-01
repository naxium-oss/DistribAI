"""Tests for CORS default policy."""

from __future__ import annotations

from services_python.cors_policy import cors_is_permissive, cors_origins_list


def test_cors_defaults_to_localhost_dashboards(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    origins = cors_origins_list()
    assert "http://127.0.0.1:3210" in origins
    assert "*" not in origins
    assert cors_is_permissive() is False


def test_cors_explicit_wildcard_is_permissive(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    assert cors_is_permissive() is True
    assert cors_origins_list() == ["*"]
