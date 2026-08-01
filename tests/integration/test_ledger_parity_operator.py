"""Integration: operator status exposes SQL vs in-memory ledger parity."""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from services_python.constants import SIGNING_KEY
from services_python.db_manager import DBManager
from services_python.orchestrator_grpc import NodeService, _make_admin_app
from worker.src.daemon.credit_ledger import CreditLedger


@pytest.fixture
async def operator_app(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")
    monkeypatch.setenv("SIGNING_KEY", "parity-test-signing-key")
    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "parity.db"), str(schema_path))
    svc = NodeService(db)
    app = _make_admin_app(svc)
    try:
        yield svc, app
    finally:
        await svc.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_operator_status_no_drift_after_in_process_transfer(operator_app):
    svc, app = operator_app
    svc.record_credit_earn("node-a", 40.0)
    result = await svc.credit_transfers.transfer("node-a", "node-b", 10.0, "parity-op")
    assert result.get("ledger_mirrored") is True

    client = TestClient(TestServer(app))
    async with client:
        resp = await client.get("/api/operator/status")
        assert resp.status == 200
        body = await resp.json()
        assert body["ledger_chain_ok"] is True
        assert body["ledger_sql_memory_drift_count"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ledger_replay_heals_after_in_memory_ledger_reset(operator_app):
    """Empty signed ledger is healed by replaying SQL rows (orchestrator restart path)."""
    svc, _app = operator_app
    svc.record_credit_earn("node-a", 25.0)
    await svc.credit_transfers.transfer("node-a", "node-b", 5.0, "reopen-test")
    assert svc.ledger_parity_summary()["drift_count"] == 0

    svc.credit_ledger = CreditLedger(signing_key=SIGNING_KEY)
    assert svc.ledger_parity_summary()["drift_count"] >= 1

    svc._replay_signed_ledger_from_sql()
    summary = svc.ledger_parity_summary()
    assert summary["chain_ok"] is True
    assert summary["drift_count"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_node_service_replays_sql_on_startup(monkeypatch, tmp_path):
    """New NodeService process reloads signed ledger from SQLite credit_ledger table."""
    monkeypatch.setenv("SIGNING_KEY", "replay-startup-key")
    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db_path = tmp_path / "replay-startup.db"
    db = DBManager(str(db_path), str(schema_path))
    svc = NodeService(db)
    svc.record_credit_earn("node-a", 30.0)
    await svc.credit_transfers.transfer("node-a", "node-b", 8.0, "startup-replay")
    await svc.close()

    db2 = DBManager(str(db_path), str(schema_path))
    svc2 = NodeService(db2)
    try:
        summary = svc2.ledger_parity_summary()
        assert summary["drift_count"] == 0
        assert summary["memory_records"] >= 3
        assert summary["chain_ok"] is True
    finally:
        await svc2.close()
