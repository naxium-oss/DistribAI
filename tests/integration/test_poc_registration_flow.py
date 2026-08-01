"""Integration: PoC challenge → enhanced register on live admin app."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from services_python.db_manager import DBManager
from services_python.orchestrator_grpc import NodeService, _make_admin_app


@pytest.fixture
async def poc_app(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RATE_LIMIT_DISABLED", "1")

    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "poc.db"), str(schema_path))
    node_service = NodeService(db)
    app = _make_admin_app(node_service)
    try:
        yield app
    finally:
        await node_service.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_enhanced_register_after_challenge(poc_app):
    client = TestClient(TestServer(poc_app))
    async with client:
        challenge_resp = await client.post("/v1/nodes/challenge")
        assert challenge_resp.status == 200
        challenge = await challenge_resp.json()

        reg_resp = await client.post(
            "/v1/nodes/register-enhanced",
            data=json.dumps(
                {
                    "node_id": "poc-node-01",
                    "challenge_id": challenge["challenge_id"],
                    "nonce": "0",
                    "os": "linux",
                    "gpu_model": "cpu",
                    "vram_mb": 0,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        assert reg_resp.status in (200, 201, 403)
        if reg_resp.status == 403:
            body = await reg_resp.json()
            assert body.get("error") in ("challenge verification failed", "sybil check failed")
