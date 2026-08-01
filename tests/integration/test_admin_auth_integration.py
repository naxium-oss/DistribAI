"""Integration tests for admin auth on the real admin app factory."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from services_python.db_manager import DBManager
from services_python.orchestrator_grpc import NodeService, _make_admin_app


@pytest.fixture
async def admin_app_live(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")
    monkeypatch.delenv("ADMIN_REQUIRE_AUTH", raising=False)

    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db_path = tmp_path / "integration.db"
    db = DBManager(str(db_path), str(schema_path))
    node_service = NodeService(db)
    app = _make_admin_app(node_service)
    try:
        yield app
    finally:
        await node_service.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_admin_app_health_open_on_loopback(admin_app_live):
    client = TestClient(TestServer(admin_app_live))
    async with client:
        resp = await client.get("/admin/health")
        assert resp.status == 200
        body = await resp.json()
        assert body.get("ok") is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_admin_app_jobs_require_auth_when_forced(admin_app_live, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "integration-secret")

    client = TestClient(TestServer(admin_app_live))
    async with client:
        denied = await client.get("/admin/jobs")
        assert denied.status == 401
        allowed = await client.get(
            "/admin/jobs",
            headers={"Authorization": "Bearer integration-secret"},
        )
        assert allowed.status == 200


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_admin_stream_requires_auth_when_forced(admin_app_live, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "integration-secret")

    client = TestClient(TestServer(admin_app_live))
    async with client:
        denied = await client.get("/admin/stream")
        assert denied.status == 401


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_admin_stream_returns_429_when_limit_exceeded(admin_app_live, monkeypatch):
    from services_python.sse_limits import admin_sse_limiter

    monkeypatch.setenv("ADMIN_SSE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("ADMIN_SSE_MAX_PER_IP", "1")
    limiter = admin_sse_limiter()
    assert await limiter.try_acquire("127.0.0.1") is True
    try:
        client = TestClient(TestServer(admin_app_live))
        async with client:
            resp = await client.get("/admin/stream")
            assert resp.status == 429
            body = await resp.json()
            assert body["error"] == "too_many_sse_connections"
    finally:
        await limiter.release("127.0.0.1")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_admin_metrics_require_auth_when_forced(admin_app_live, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "integration-secret")

    client = TestClient(TestServer(admin_app_live))
    async with client:
        for path in ("/admin/metrics/system", "/admin/metrics/orchestrator"):
            denied = await client.get(path)
            assert denied.status == 401, path
            allowed = await client.get(
                path,
                headers={"Authorization": "Bearer integration-secret"},
            )
            assert allowed.status == 200, path
