"""NodeService.record_credit_earn keeps SQL and signed ledger aligned."""

from __future__ import annotations

from pathlib import Path

import pytest

from services_python.db_manager import DBManager
from services_python.orchestrator_grpc import NodeService


@pytest.fixture
async def earn_service(tmp_path):
    schema = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "earn.db"), str(schema))
    service = NodeService(db)
    try:
        yield service
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_record_credit_earn_updates_sql_and_signed_ledger(earn_service):
    svc = earn_service
    svc.record_credit_earn("earn-node-2", 6.0, "job-99", {"task_id": "task-1"})

    row = svc.db.get_node_credits("earn-node-2")
    assert row["balance"] == 6.0
    assert svc.credit_ledger.get_balance("earn-node-2") == pytest.approx(6.0)
    assert svc.credit_ledger.verify_chain_integrity()
