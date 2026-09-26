"""Integration: in-memory signed ledger mirrors SQL after transfers."""

from __future__ import annotations

from pathlib import Path

import pytest

from services_python.db_manager import DBManager
from services_python.orchestrator_grpc import NodeService


@pytest.fixture
async def node_service_ledger(tmp_path):
    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "ledger-mem.db"), str(schema_path))
    service = NodeService(db)
    try:
        yield service
    finally:
        await service.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_transfer_mirrors_signed_ledger_and_sql(node_service_ledger):
    svc = node_service_ledger
    svc.db.add_credits("ledger-a", 50.0)
    before_size = svc.credit_ledger.size()

    result = await svc.credit_transfers.transfer("ledger-a", "ledger-b", 12.5, "parity-test")
    assert result["success"] is True
    assert result.get("ledger_mirrored") is True

    a = svc.db.get_node_credits("ledger-a")
    b = svc.db.get_node_credits("ledger-b")
    assert a["balance"] == 37.5
    assert b["balance"] == 12.5

    assert svc.credit_ledger.size() == before_size + 2
    assert svc.credit_ledger.verify_chain_integrity()

    mem_a = svc.credit_ledger.get_balance("ledger-a")
    mem_b = svc.credit_ledger.get_balance("ledger-b")
    assert mem_a == pytest.approx(-12.5)
    assert mem_b == pytest.approx(12.5)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_task_earn_updates_both_ledgers(node_service_ledger):
    svc = node_service_ledger
    svc.credit_ledger.credit("earn-node", 8.0, "job-1")
    svc.db.add_credits("earn-node", 8.0, tx_type="earn", metadata={"job_id": "job-1"})

    row = svc.db.get_node_credits("earn-node")
    assert row["balance"] == 8.0
    assert svc.credit_ledger.get_balance("earn-node") == pytest.approx(8.0)
    assert svc.credit_ledger.verify_chain_integrity()
