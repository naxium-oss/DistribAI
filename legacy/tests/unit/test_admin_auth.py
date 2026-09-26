"""Unit tests for admin HTTP authentication middleware."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from services_python.admin_auth import (
    admin_auth_enforced,
    admin_auth_middleware,
    resolve_admin_secret,
    validate_production_startup,
)


@pytest.fixture
def admin_env(monkeypatch):
    monkeypatch.delenv("ADMIN_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("DISTRIBAI_ADMIN_SECRET", raising=False)
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")
    monkeypatch.setenv("JWT_SECRET", "test-jwt-secret")


def test_admin_auth_not_enforced_on_loopback_by_default(admin_env):
    assert admin_auth_enforced() is False


def test_admin_auth_enforced_when_admin_host_public(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_HOST", "0.0.0.0")
    assert admin_auth_enforced() is True


def test_admin_auth_enforced_when_explicit(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    assert admin_auth_enforced() is True


def test_resolve_admin_secret_prefers_dedicated(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "dedicated-secret")
    assert resolve_admin_secret() == "dedicated-secret"


def test_resolve_admin_secret_requires_dedicated_when_enforced(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.delenv("DISTRIBAI_ADMIN_SECRET", raising=False)
    assert resolve_admin_secret() is None


async def _make_client(monkeypatch, handler_body: str = "ok") -> TestClient:
    async def ok_handler(_request: web.Request) -> web.Response:
        return web.Response(text=handler_body)

    app = web.Application(middlewares=[admin_auth_middleware])
    app.router.add_get("/admin/nodes", ok_handler)
    app.router.add_get("/admin/health", ok_handler)
    return TestClient(TestServer(app))


@pytest.mark.asyncio
async def test_middleware_allows_open_localhost(admin_env):
    client = await _make_client(admin_env)
    async with client:
        resp = await client.get("/admin/nodes")
        assert resp.status == 200


@pytest.mark.asyncio
async def test_middleware_returns_503_when_enforced_without_secret(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.delenv("DISTRIBAI_ADMIN_SECRET", raising=False)
    client = await _make_client(admin_env)
    async with client:
        resp = await client.get("/admin/nodes")
        assert resp.status == 503


@pytest.mark.asyncio
async def test_middleware_blocks_without_token_when_enforced(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "admin-token")
    client = await _make_client(admin_env)
    async with client:
        resp = await client.get("/admin/nodes")
        assert resp.status == 401


@pytest.mark.asyncio
async def test_middleware_allows_bearer_when_enforced(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "admin-token")
    client = await _make_client(admin_env)
    async with client:
        resp = await client.get(
            "/admin/nodes",
            headers={"Authorization": "Bearer admin-token"},
        )
        assert resp.status == 200


@pytest.mark.asyncio
async def test_public_release_publish_requires_auth_when_enforced(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "admin-token")

    async def ok_handler(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    app = web.Application(middlewares=[admin_auth_middleware])
    app.router.add_post("/api/admin/public-release/publish", ok_handler)
    client = TestClient(TestServer(app))
    async with client:
        resp = await client.post("/api/admin/public-release/publish", json={"push": True})
        assert resp.status == 401


@pytest.mark.asyncio
async def test_api_admin_prefix_requires_auth_when_enforced(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "admin-token")

    async def ok_handler(_request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application(middlewares=[admin_auth_middleware])
    app.router.add_post("/api/admin/distribai/registry/sync", ok_handler)
    client = TestClient(TestServer(app))
    async with client:
        denied = await client.post("/api/admin/distribai/registry/sync", json={})
        assert denied.status == 401
        allowed = await client.post(
            "/api/admin/distribai/registry/sync",
            json={},
            headers={"Authorization": "Bearer admin-token"},
        )
        assert allowed.status == 200


@pytest.mark.asyncio
async def test_public_bind_blocks_anonymous_jobs_list(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_HOST", "0.0.0.0")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "remote-secret")

    async def ok_handler(_request: web.Request) -> web.Response:
        return web.json_response({"jobs": []})

    app = web.Application(middlewares=[admin_auth_middleware])
    app.router.add_get("/admin/jobs", ok_handler)
    client = TestClient(TestServer(app))
    async with client:
        resp = await client.get("/admin/jobs")
        assert resp.status == 401


def test_validate_startup_rejects_public_without_secrets(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_HOST", "0.0.0.0")
    monkeypatch.delenv("SIGNING_KEY", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("ALLOW_INSECURE_PUBLIC_BIND", raising=False)
    with pytest.raises(SystemExit) as exc:
        validate_production_startup()
    assert exc.value.code == 1


def test_validate_startup_allows_public_with_opt_out(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_HOST", "0.0.0.0")
    monkeypatch.delenv("SIGNING_KEY", raising=False)
    monkeypatch.setenv("ALLOW_INSECURE_PUBLIC_BIND", "1")
    validate_production_startup()


@pytest.mark.asyncio
async def test_admin_stream_requires_auth_when_enforced(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "admin-token")

    async def stream_handler(_request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(status=200)
        await response.prepare(_request)
        return response

    app = web.Application(middlewares=[admin_auth_middleware])
    app.router.add_get("/admin/stream", stream_handler)
    client = TestClient(TestServer(app))
    async with client:
        resp = await client.get("/admin/stream")
        assert resp.status == 401


@pytest.mark.asyncio
async def test_admin_metrics_requires_auth_when_enforced(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "admin-token")

    async def metrics_handler(_request: web.Request) -> web.Response:
        return web.json_response({"cpu": 1})

    app = web.Application(middlewares=[admin_auth_middleware])
    app.router.add_get("/admin/metrics/system", metrics_handler)
    client = TestClient(TestServer(app))
    async with client:
        resp = await client.get("/admin/metrics/system")
        assert resp.status == 401


def test_admin_paginated_routes_registered_before_id_capture():
    """Regression: /admin/jobs/paginated must not match {job_id}='paginated'."""
    source = (
        Path(__file__).resolve().parents[2] / "services_python" / "orchestrator_grpc.py"
    ).read_text(encoding="utf-8")
    jobs_paginated = source.index('"/admin/jobs/paginated"')
    jobs_id = source.index('"/admin/jobs/{job_id}"')
    credits_paginated = source.index('"/admin/credits/paginated"')
    credits_id = source.index('"/admin/credits/{node_id}"')
    assert jobs_paginated < jobs_id
    assert credits_paginated < credits_id


@pytest.mark.asyncio
async def test_api_docs_requires_auth_when_enforced(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "admin-token")

    async def ok_handler(_request: web.Request) -> web.Response:
        return web.json_response([])

    app = web.Application(middlewares=[admin_auth_middleware])
    app.router.add_get("/api/docs/list", ok_handler)
    client = TestClient(TestServer(app))
    async with client:
        denied = await client.get("/api/docs/list")
        assert denied.status == 401
        allowed = await client.get(
            "/api/docs/list",
            headers={"Authorization": "Bearer admin-token"},
        )
        assert allowed.status == 200


@pytest.mark.asyncio
async def test_health_exempt_even_when_enforced(admin_env, monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    client = await _make_client(admin_env)
    async with client:
        resp = await client.get("/admin/health")
        assert resp.status == 200
