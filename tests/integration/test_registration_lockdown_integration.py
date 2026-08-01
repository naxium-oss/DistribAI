"""Integration: open registration blocked when lockdown is on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from services_python.db_manager import DBManager
from services_python.orchestrator_grpc import NodeService, _make_admin_app


@pytest.fixture
async def admin_app_lockdown(monkeypatch, tmp_path):
    monkeypatch.setenv("REGISTRATION_REQUIRE_POC", "true")
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "integration-secret")
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")

    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "reg.db"), str(schema_path))
    node_service = NodeService(db)
    app = _make_admin_app(node_service)
    try:
        yield app
    finally:
        await node_service.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_open_register_returns_403_when_poc_required(admin_app_lockdown):
    client = TestClient(TestServer(admin_app_lockdown))
    async with client:
        resp = await client.post(
            "/v1/nodes/register",
            data=json.dumps(
                {"node_id": "integration-node-01", "os": "linux", "gpu_model": "cpu", "vram_mb": 0}
            ),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 403
        body = await resp.json()
        assert body["error"] == "registration_requires_poc"
