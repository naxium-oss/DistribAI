"""Credit ledger mixin with a tamper-evident per-node hash chain."""

import hashlib
import json
import time
from typing import Any


def _chain_hash(
    prev_hash: str,
    node_id: str,
    tx_type: str,
    amount: float,
    balance_after: float,
    ts: int,
) -> str:
    """SHA-256 chain hash linking this credit row to the node's previous one.

    Chaining ``prev_hash`` into each digest makes the SQL ledger
    tamper-evident: editing or deleting any historical row invalidates every
    subsequent ``tx_hash`` for that node, which :meth:`verify_credit_chain`
    detects.
    """
    payload = json.dumps(
        {
            "prev": prev_hash,
            "node": node_id,
            "type": tx_type,
            "amount": round(float(amount), 8),
            "balance_after": round(float(balance_after), 8),
            "ts": int(ts),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CreditsMixin:
    """Mixin for DBManager."""

    def _latest_chain_hash(self, conn, node_id: str) -> str:
        """Most recent ``tx_hash`` for a node ("genesis" when none yet)."""
        row = conn.execute(
            """
            SELECT tx_hash FROM credit_ledger
            WHERE node_id = ? AND tx_hash IS NOT NULL
            ORDER BY tx_id DESC LIMIT 1
            """,
            (node_id,),
        ).fetchone()
        return (row["tx_hash"] if row and row["tx_hash"] else "genesis")

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
            prev_hash = self._latest_chain_hash(conn, node_id)
            tx_hash = _chain_hash(prev_hash, node_id, tx_type, float(amount), new_balance, now)
            # Preserve caller metadata but never let it override the chain fields.
            meta = {k: v for k, v in metadata.items() if k not in {"tx_hash", "prev_hash"}}
            cursor = conn.execute(
                """
                INSERT INTO credit_ledger
                    (node_id, tx_type, amount, balance_after, tx_hash, prev_hash, ts, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    tx_type,
                    float(amount),
                    new_balance,
                    tx_hash,
                    prev_hash,
                    now,
                    json.dumps(meta) if meta else None,
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
                meta = json.dumps({"transfer_id": tx_tag, "memo": note})
                out_prev = self._latest_chain_hash(conn, from_node_id)
                out_hash = _chain_hash(
                    out_prev, from_node_id, "transfer_out", -amount_f, new_from, now
                )
                conn.execute(
                    """
                INSERT INTO credit_ledger
                    (node_id, tx_type, amount, balance_after, tx_hash, prev_hash, ts, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (from_node_id, "transfer_out", -amount_f, new_from, out_hash, out_prev, now, meta),
                )
                balance_row_b = conn.execute(
                    "SELECT COALESCE(SUM(amount), 0) AS balance FROM credit_ledger WHERE node_id = ?",
                    (to_node_id,),
                ).fetchone()
                new_to = float(balance_row_b["balance"] or 0) + amount_f
                in_prev = self._latest_chain_hash(conn, to_node_id)
                in_hash = _chain_hash(in_prev, to_node_id, "transfer_in", amount_f, new_to, now)
                conn.execute(
                    """
                INSERT INTO credit_ledger
                    (node_id, tx_type, amount, balance_after, tx_hash, prev_hash, ts, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (to_node_id, "transfer_in", amount_f, new_to, in_hash, in_prev, now, meta),
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
                SELECT tx_id, node_id, tx_type, amount, tx_hash, prev_hash, ts, metadata_json
                FROM credit_ledger
                ORDER BY tx_id ASC
                """
            )
            return [dict(row) for row in cur.fetchall()]

    def verify_credit_chain(self) -> dict[str, Any]:
        """Recompute the per-node SQL hash chain and report any breakage.

        Returns a summary with ``ok`` (bool), the number of rows checked, and
        up to a few offending ``tx_id`` values. Legacy rows written before
        chaining existed (``tx_hash IS NULL``) are skipped, not failed.
        """
        broken: list[int] = []
        checked = 0
        last_hash: dict[str, str] = {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT tx_id, node_id, tx_type, amount, balance_after, tx_hash, prev_hash, ts
                FROM credit_ledger
                ORDER BY tx_id ASC
                """
            ).fetchall()
        for row in rows:
            node_id = row["node_id"]
            if row["tx_hash"] is None:
                # Pre-chain legacy row; reset the expected predecessor.
                last_hash[node_id] = row["tx_hash"] or last_hash.get(node_id, "genesis")
                continue
            checked += 1
            expected_prev = last_hash.get(node_id, "genesis")
            recomputed = _chain_hash(
                row["prev_hash"] or "genesis",
                node_id,
                row["tx_type"],
                row["amount"],
                row["balance_after"],
                row["ts"],
            )
            if row["prev_hash"] != expected_prev or recomputed != row["tx_hash"]:
                broken.append(int(row["tx_id"]))
            last_hash[node_id] = row["tx_hash"]
        return {"ok": not broken, "checked": checked, "broken_tx_ids": broken[:10]}

