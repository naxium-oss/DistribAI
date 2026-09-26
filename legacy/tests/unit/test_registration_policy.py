"""Tests for node registration policy."""

from __future__ import annotations

from services_python.registration_policy import registration_requires_poc


def test_registration_open_on_loopback_by_default(monkeypatch):
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")
    monkeypatch.delenv("REGISTRATION_REQUIRE_POC", raising=False)
    assert registration_requires_poc() is False


def test_registration_independent_of_public_admin_bind(monkeypatch):
    monkeypatch.setenv("ADMIN_HOST", "0.0.0.0")
    monkeypatch.delenv("REGISTRATION_REQUIRE_POC", raising=False)
    assert registration_requires_poc() is False


def test_registration_explicit_override(monkeypatch):
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")
    monkeypatch.setenv("REGISTRATION_REQUIRE_POC", "true")
    assert registration_requires_poc() is True
