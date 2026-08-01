"""Coverage harness: critical admin routes must be registered on the live app factory."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from services_python.db_manager import DBManager
from services_python.orchestrator_grpc import NodeService, _make_admin_app

_CRITICAL_GET_PATHS = (
    "/admin/health",
    "/admin/jobs",
    "/admin/nodes",
    "/admin/credits",
    "/admin/stream",
    "/admin/metrics/system",
    "/admin/ledger/root",
    "/api/operator/status",
    "/api/docs/list",
    "/v1/queue",
)


@pytest.fixture
async def route_registry_app(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")
    monkeypatch.delenv("ADMIN_REQUIRE_AUTH", raising=False)
    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "routes.db"), str(schema_path))
    node_service = NodeService(db)
    app = _make_admin_app(node_service)
    try:
        yield app
    finally:
        await node_service.close()


@pytest.mark.asyncio
async def test_critical_get_routes_are_not_404(route_registry_app):
    client = TestClient(TestServer(route_registry_app))
    async with client:
        for path in _CRITICAL_GET_PATHS:
            resp = await client.get(path)
            assert resp.status != 404, f"missing route: {path} (status={resp.status})"


def _manifest_get_paths() -> list[str]:
    manifest = (
        Path(__file__).resolve().parents[2] / "scripts" / "ci" / "admin_route_manifest.txt"
    )
    paths: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        paths.append(line)
    return paths


@pytest.mark.asyncio
async def test_manifest_get_routes_are_not_404(route_registry_app):
    client = TestClient(TestServer(route_registry_app))
    async with client:
        for path in _manifest_get_paths():
            resp = await client.get(path)
            assert resp.status != 404, f"manifest route missing: {path} (status={resp.status})"


@pytest.mark.asyncio
async def test_rebenchmark_trigger_is_real_and_unsupported_routes_are_absent(route_registry_app):
    client = TestClient(TestServer(route_registry_app))
    async with client:
        response = await client.post("/api/admin/rebenchmark/trigger", json={})
        assert response.status == 200
        body = await response.json()
        assert body["ok"] is True
        assert body["scheduled"] == 0

        for path, method in (
            ("/api/admin/backup/create", "post"),
            ("/api/admin/cache/clear", "post"),
            ("/api/admin/import/jobs", "post"),
            ("/api/admin/users", "get"),
        ):
            response = await getattr(client, method)(path)
            assert response.status == 404, f"dead placeholder route still exists: {path}"
