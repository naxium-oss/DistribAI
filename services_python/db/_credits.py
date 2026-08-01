"""Credit ledger mixin."""

import time
from typing import Any


class CreditsMixin:
    """Mixin for DBManager."""

    def add_credits(
        self,
        node_id: str,
        amount: float,
        tx_type: str = "earn",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = int(time.time())
        metadata = metadata or {}
        with self._connect() as conn:
            balance_row = conn.execute(
                "SELECT COALESCE(SUM(amount), 0) AS balance FROM credit_ledger WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            new_balance = float(balance_row["balance"] or 0) + float(amount)
            cursor = conn.execute(
                """
                INSERT INTO credit_ledger (node_id, tx_type, amount, balance_after, tx_hash, prev_hash, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    tx_type,
                    float(amount),
                    new_balance,
                    metadata.get("tx_hash"),
                    metadata.get("prev_hash"),
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def transfer_credits_between_nodes(
        self,
        from_node_id: str,
        to_node_id: str,
        amount: float,
        transfer_id: str,
        memo: str = "",
    ) -> None:
        """Debit ``from_node_id`` and credit ``to_node_id`` in one SQLite transaction."""
        amount_f = float(amount)
        if amount_f <= 0:
            raise ValueError("transfer amount must be positive")
        now = int(time.time())
        tx_tag = (transfer_id or "")[:128]
        note = (memo or "")[:512]
        with self._conn_lock:
            conn = self._ensure_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                balance_row = conn.execute(
                    "SELECT COALESCE(SUM(amount), 0) AS balance FROM credit_ledger WHERE node_id = ?",
                    (from_node_id,),
                ).fetchone()
                from_bal = float(balance_row["balance"] or 0)
                if from_bal < amount_f:
                    raise ValueError("Insufficient credits for transfer")
                new_from = from_bal - amount_f
                conn.execute(
                    """
                INSERT INTO credit_ledger (node_id, tx_type, amount, balance_after, tx_hash, prev_hash, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (from_node_id, "transfer_out", -amount_f, new_from, tx_tag, note, now),
                )
                balance_row_b = conn.execute(
                    "SELECT COALESCE(SUM(amount), 0) AS balance FROM credit_ledger WHERE node_id = ?",
                    (to_node_id,),
                ).fetchone()
                new_to = float(balance_row_b["balance"] or 0) + amount_f
                conn.execute(
                    """
                INSERT INTO credit_ledger (node_id, tx_type, amount, balance_after, tx_hash, prev_hash, ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (to_node_id, "transfer_in", amount_f, new_to, tx_tag, note, now),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def get_node_credits(self, node_id: str) -> dict[str, float] | None:
        with self._conn_lock:
            conn = self._ensure_conn()
            row = conn.execute(
                """
            SELECT node_id,
                   COALESCE(SUM(amount), 0) AS balance,
                   COALESCE(SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END), 0) AS lifetime,
                   COALESCE(ABS(SUM(CASE WHEN tx_type = 'vote' THEN amount ELSE 0 END)), 0) AS votes_cast
            FROM credit_ledger
            WHERE node_id = ?
            """,
                (node_id,),
            ).fetchone()
        if row and row["node_id"]:
            return {
                "balance": float(row["balance"] or 0),
                "lifetime": float(row["lifetime"] or 0),
                "votes_cast": float(row["votes_cast"] or 0),
            }
        return None

    def list_all_credits(self) -> dict[str, dict[str, float]]:
        with self._conn_lock:
            conn = self._ensure_conn()
            cur = conn.execute(
                """
            SELECT node_id,
                   COALESCE(SUM(amount), 0) AS balance,
                   COALESCE(SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END), 0) AS lifetime,
                   COALESCE(ABS(SUM(CASE WHEN tx_type = 'vote' THEN amount ELSE 0 END)), 0) AS votes_cast
            FROM credit_ledger
            GROUP BY node_id
            """
            )
            rows = cur.fetchall()
        return {
            row["node_id"]: {
                "balance": float(row["balance"] or 0),
                "lifetime": float(row["lifetime"] or 0),
                "votes_cast": float(row["votes_cast"] or 0),
            }
            for row in rows
        }

    def iter_credit_ledger_rows(self) -> list[dict[str, Any]]:
        """Return credit_ledger SQL rows in tx_id order for signed-ledger replay."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT tx_id, node_id, tx_type, amount, tx_hash, prev_hash, ts
                FROM credit_ledger
                ORDER BY tx_id ASC
                """
            )
            return [dict(row) for row in cur.fetchall()]

