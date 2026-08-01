"""Tests for CreditTransferManager.transfer persistence."""

from pathlib import Path

import pytest

from services_python.credit_transfers import CreditTransferManager
from services_python.db_manager import DBManager
from worker.src.daemon.credit_ledger import CreditLedger


@pytest.fixture
def memory_db():
    root = Path(__file__).resolve().parents[2]
    schema = root / "runtime" / "db" / "schema.sql"
    return DBManager(":memory:", str(schema))


@pytest.mark.asyncio
async def test_transfer_persists_balances(memory_db):
    mgr = CreditTransferManager(memory_db)
    memory_db.add_credits("alice", 500.0, "earn", {})
    memory_db.add_credits("bob", 10.0, "earn", {})

    result = await mgr.transfer("alice", "bob", 100.0, "pooling")

    assert result["success"] is True
    assert result["transfer_id"]
    assert result.get("ledger_mirrored") is False
    alice = memory_db.get_node_credits("alice")
    bob = memory_db.get_node_credits("bob")
    assert alice is not None and bob is not None
    assert alice["balance"] == 400.0
    assert bob["balance"] == 110.0


@pytest.mark.asyncio
async def test_transfer_mirrors_signed_ledger_when_configured(memory_db):
    ledger = CreditLedger(signing_key=b"k" * 32)
    ledger.credit("alice", 500.0, job_id="seed")
    ledger.credit("bob", 10.0, job_id="seed")
    mgr = CreditTransferManager(memory_db, credit_ledger=ledger)
    memory_db.add_credits("alice", 500.0, "earn", {})
    memory_db.add_credits("bob", 10.0, "earn", {})

    result = await mgr.transfer("alice", "bob", 100.0, "pooling")

    assert result["success"] is True
    assert result.get("ledger_mirrored") is True
    assert ledger.get_balance("alice") == 400.0
    assert ledger.get_balance("bob") == 110.0
    assert ledger.verify_chain_integrity()


@pytest.mark.asyncio
async def test_transfer_insufficient_balance(memory_db):
    mgr = CreditTransferManager(memory_db)
    memory_db.add_credits("poor", 5.0, "earn", {})

    result = await mgr.transfer("poor", "rich", 100.0, "x")

    assert result["success"] is False
    assert "Insufficient" in (result.get("error") or "")
    assert memory_db.get_node_credits("rich") is None


@pytest.mark.asyncio
async def test_transfer_requires_database():
    mgr = CreditTransferManager(None)
    result = await mgr.transfer("a", "b", 1.0, "")
    assert result["success"] is False
