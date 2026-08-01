"""Health endpoint exposes job submission availability."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from services_python.db_manager import DBManager
from services_python.orchestrator_grpc import NodeService, _make_admin_app


@pytest.fixture
async def health_app(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")
    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "health.db"), str(schema_path))
    node_service = NodeService(db)
    app = _make_admin_app(node_service)
    try:
        yield app
    finally:
        await node_service.close()


@pytest.mark.asyncio
async def test_health_includes_job_submission_flag(health_app):
    client = TestClient(TestServer(health_app))
    async with client:
        resp = await client.get("/admin/health")
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        assert "job_submission_available" in body
        assert isinstance(body["job_submission_available"], bool)
