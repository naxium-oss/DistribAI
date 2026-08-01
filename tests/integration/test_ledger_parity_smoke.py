"""Integration smoke: SQL credit balances match ledger row sums."""

from __future__ import annotations

from pathlib import Path

import pytest

from services_python.db_manager import DBManager


@pytest.fixture
def ledger_db(tmp_path):
    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    return DBManager(str(tmp_path / "ledger.db"), str(schema_path))


@pytest.mark.integration
def test_credit_balance_matches_ledger_sum(ledger_db):
    node_id = "parity-node-01"
    ledger_db.add_credits(node_id, 10.0, tx_type="earn")
    ledger_db.add_credits(node_id, 5.0, tx_type="earn")

    summary = ledger_db.get_node_credits(node_id)
    assert summary is not None
    assert summary["balance"] == 15.0

    with ledger_db._connect() as conn:
        rows = conn.execute(
            "SELECT amount, balance_after FROM credit_ledger WHERE node_id = ? ORDER BY tx_id",
            (node_id,),
        ).fetchall()

    assert sum(float(row["amount"]) for row in rows) == 15.0
    assert float(rows[-1]["balance_after"]) == 15.0


@pytest.mark.integration
def test_transfer_preserves_total_credits(ledger_db):
    ledger_db.add_credits("node-a", 20.0)
    ledger_db.transfer_credits_between_nodes("node-a", "node-b", 7.5, "xfer-1", "test")

    a = ledger_db.get_node_credits("node-a")
    b = ledger_db.get_node_credits("node-b")
    assert a["balance"] == 12.5
    assert b["balance"] == 7.5
