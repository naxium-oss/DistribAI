"""Tests for public operator status disclosure endpoint."""

from __future__ import annotations

import pytest
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_operator_status_returns_posture_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_HOST", "0.0.0.0")
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("REGISTRATION_REQUIRE_POC", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "secret")
    monkeypatch.setenv("SIGNING_KEY", "")
    monkeypatch.setenv("JWT_SECRET", "")

    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: None)

    from pathlib import Path

    from services_python.db_manager import DBManager
    from services_python.orchestrator_grpc import NodeService, _make_admin_app

    schema = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "op.db"), str(schema))
    ns = NodeService(db)
    app = _make_admin_app(ns)

    client = TestClient(TestServer(app))
    async with client:
        resp = await client.get("/api/operator/status")
        assert resp.status == 200
        body = await resp.json()
        assert body["admin_auth_enforced"] is True
        assert body["registration_requires_poc"] is True
        assert body["admin_host"] == "0.0.0.0"
        assert body["signing_key_from_env"] is False

    await ns.close()
