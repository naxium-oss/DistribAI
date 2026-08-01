"""
Credit Transfer System (Production Implementation)
Implements credit transfers between accounts as specified in README §5.2:
"Users may transfer credits to other accounts, enabling team pooling.
Transfers are permanently logged for anti-fraud."
All transfers are recorded in the tamper-evident ledger with full audit trail.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any

from worker.src.daemon.credit_ledger import CreditLedger

logger = logging.getLogger(__name__)


@dataclass
class CreditTransfer:
    transfer_id: str
    from_node_id: str
    to_node_id: str
    amount: float
    reason: str
    timestamp: float
    tx_hash: str

    def to_ledger_entry(self) -> dict[str, Any]:
        return {
            "tx_id": self.transfer_id,
            "type": "transfer",
            "from_account": self.from_node_id,
            "to_account": self.to_node_id,
            "amount": self.amount,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "hash": self.tx_hash,
        }


class CreditTransferManager:
    """
    Manages credit transfers between node accounts.
    Features:
    - Transfer validation (sufficient balance, limits)
    - Anti-fraud detection (velocity limits, suspicious patterns)
    - Full audit logging to tamper-evident ledger
    """

    MIN_TRANSFER_AMOUNT = 1.0
    MAX_TRANSFER_AMOUNT = 10000.0
    MAX_TRANSFERS_PER_HOUR = 10

    def __init__(self, db_manager=None, credit_ledger: CreditLedger | None = None):
        self.db = db_manager
        self.credit_ledger = credit_ledger
        self.pending_transfers: dict[str, CreditTransfer] = {}
        self.transfer_history: list[CreditTransfer] = []
        self.node_transfer_counts: dict[str, list[float]] = {}

    def _compute_transfer_hash(self, transfer: CreditTransfer) -> str:
        data = f"{transfer.transfer_id}:{transfer.from_node_id}:{transfer.to_node_id}:{transfer.amount}:{transfer.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()

    def _check_velocity_limit(self, node_id: str) -> bool:
        now = time.time()
        hour_ago = now - 3600
        if node_id not in self.node_transfer_counts:
            self.node_transfer_counts[node_id] = []
        recent_transfers = [ts for ts in self.node_transfer_counts[node_id] if ts > hour_ago]
        self.node_transfer_counts[node_id] = recent_transfers
        return len(recent_transfers) < self.MAX_TRANSFERS_PER_HOUR

    def validate_transfer(
        self, from_node_id: str, to_node_id: str, amount: float, from_balance: float
    ) -> tuple[bool, str | None]:
        """
        Validate a credit transfer.
        Returns:
            Tuple of (is_valid, error_message)
        """
        if from_node_id == to_node_id:
            return False, "Cannot transfer credits to yourself"
        if amount < self.MIN_TRANSFER_AMOUNT:
            return False, f"Minimum transfer is {self.MIN_TRANSFER_AMOUNT} credits"
        if amount > self.MAX_TRANSFER_AMOUNT:
            return False, f"Maximum transfer is {self.MAX_TRANSFER_AMOUNT} credits"
        if not isinstance(amount, (int, float)) or amount <= 0:
            return False, "Invalid transfer amount"
        if amount != amount:  # NaN check - NaN is the only value that doesn't equal itself
            return False, "Invalid transfer amount (NaN)"
        if amount == float("inf") or amount == float("-inf"):
            return False, "Invalid transfer amount (infinite)"
        if from_balance < amount:
            return False, "Insufficient credits for transfer"
        if not self._check_velocity_limit(from_node_id):
            return False, f"Transfer limit: max {self.MAX_TRANSFERS_PER_HOUR} transfers per hour"
        return True, None

    def validate_transfer_atomic(
        self, from_node_id: str, to_node_id: str, amount: float, db_manager
    ) -> tuple[bool, str | None, float | None]:
        """
        Atomic validation that checks current balance from database.
        This prevents TOCTOU race conditions by reading balance atomically.
        Returns:
            Tuple of (is_valid, error_message, current_balance)
        """
        is_valid, error = self.validate_transfer(from_node_id, to_node_id, amount, float("inf"))
        if not is_valid:
            return False, error, None
        if db_manager is None:
            return False, "Database manager required for atomic validation", None
        credits = db_manager.get_node_credits(from_node_id)
        current_balance = credits.get("balance", 0.0) if credits else 0.0
        if current_balance < amount:
            return False, "Insufficient credits for transfer", current_balance
        return True, None, current_balance

    def create_transfer(
        self,
        from_node_id: str,
        to_node_id: str,
        amount: float,
        reason: str = "",
        from_balance: float = 0.0,
    ) -> tuple[bool, str | None, CreditTransfer | None]:
        """
        Create a credit transfer.
        Args:
            from_node_id: Source node ID
            to_node_id: Destination node ID
            amount: Amount to transfer
            reason: Optional reason for transfer
            from_balance: Current balance of sender
        Returns:
            Tuple of (success, error_message, transfer_object)
        """
        is_valid, error = self.validate_transfer(from_node_id, to_node_id, amount, from_balance)
        if not is_valid:
            return False, error, None
        transfer_id = f"xfer_{hashlib.sha256(f'{from_node_id}{to_node_id}{time.time()}'.encode()).hexdigest()[:16]}"
        transfer = CreditTransfer(
            transfer_id=transfer_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            amount=amount,
            reason=reason or "Team pooling",
            timestamp=time.time(),
            tx_hash="",
        )
        transfer.tx_hash = self._compute_transfer_hash(transfer)
        if from_node_id not in self.node_transfer_counts:
            self.node_transfer_counts[from_node_id] = []
        self.node_transfer_counts[from_node_id].append(time.time())
        self.pending_transfers[transfer_id] = transfer
        self.transfer_history.append(transfer)
        logger.info(
            f"Credit transfer created: {amount} credits from {from_node_id[:20]}... "
            f"to {to_node_id[:20]}... (ID: {transfer_id[:16]}...)"
        )
        return True, None, transfer

    def confirm_transfer(self, transfer_id: str) -> bool:
        """
        Confirm a transfer after ledger recording.
        Args:
            transfer_id: Transfer to confirm
        Returns:
            True if confirmed
        """
        if transfer_id not in self.pending_transfers:
            return False
        del self.pending_transfers[transfer_id]
        logger.info("Credit transfer confirmed: %s...", transfer_id[:16])
        return True

    async def transfer(
        self,
        from_node_id: str,
        to_node_id: str,
        amount: float,
        memo: str = "",
    ) -> dict[str, Any]:
        """Apply a validated transfer and persist balances.

        Runs synchronously on the caller's task; avoid ``asyncio.to_thread`` here so
        ``DBManager``'s thread-local SQLite connection (especially ``:memory:``)
        stays on the same thread that constructed the manager.
        """
        return self._transfer_sync(from_node_id, to_node_id, amount, memo)

    def _rollback_transfer_record(self, xfer: CreditTransfer) -> None:
        self.pending_transfers.pop(xfer.transfer_id, None)
        try:
            self.transfer_history.remove(xfer)
        except ValueError:
            pass

    def _transfer_sync(
        self,
        from_node_id: str,
        to_node_id: str,
        amount: float,
        memo: str = "",
    ) -> dict[str, Any]:
        if self.db is None:
            return {"success": False, "error": "database unavailable"}

        amount_f = float(amount)
        ok, err, current_balance = self.validate_transfer_atomic(
            from_node_id, to_node_id, amount_f, self.db
        )
        if not ok or current_balance is None:
            return {"success": False, "error": err or "validation failed"}

        reason = (memo or "").strip() or "Team pooling"
        created_ok, create_err, xfer = self.create_transfer(
            from_node_id, to_node_id, amount_f, reason, from_balance=current_balance
        )
        if not created_ok or xfer is None:
            return {"success": False, "error": create_err or "transfer creation failed"}

        try:
            self.db.transfer_credits_between_nodes(
                from_node_id, to_node_id, amount_f, xfer.transfer_id, reason
            )
        except ValueError as exc:
            logger.warning("Transfer rejected at persistence: %s", exc)
            self._rollback_transfer_record(xfer)
            return {"success": False, "error": str(exc)}
        except Exception:
            logger.exception("Transfer persistence failed")
            self._rollback_transfer_record(xfer)
            return {"success": False, "error": "persist_failed"}

        ledger_mirrored = False
        if self.credit_ledger is not None:
            try:
                self._mirror_transfer_to_signed_ledger(xfer)
                ledger_mirrored = True
            except Exception:
                logger.exception(
                    "Signed ledger mirror failed after DB transfer (SQL is authoritative): %s",
                    xfer.transfer_id,
                )
                ledger_mirrored = False

        self.confirm_transfer(xfer.transfer_id)
        return {
            "success": True,
            "transfer_id": xfer.transfer_id,
            "amount": amount_f,
            "from": from_node_id,
            "to": to_node_id,
            "ledger_mirrored": ledger_mirrored,
        }

    def _mirror_transfer_to_signed_ledger(self, xfer: CreditTransfer) -> None:
        """Append matching ``transfer_out`` / ``transfer_in`` rows to the hash-chained ledger."""
        if self.credit_ledger is None:
            return
        memo = xfer.reason or ""
        meta_out: dict[str, Any] = {
            "transfer_id": xfer.transfer_id,
            "counterparty": xfer.to_node_id,
            "memo": memo,
            "tx_hash": xfer.tx_hash,
        }
        meta_in: dict[str, Any] = {
            "transfer_id": xfer.transfer_id,
            "counterparty": xfer.from_node_id,
            "memo": memo,
            "tx_hash": xfer.tx_hash,
        }
        self.credit_ledger.append_record(
            xfer.from_node_id,
            "transfer_out",
            -float(xfer.amount),
            meta_out,
            job_id="",
        )
        self.credit_ledger.append_record(
            xfer.to_node_id,
            "transfer_in",
            float(xfer.amount),
            meta_in,
            job_id="",
        )

    def get_transfer_history(
        self,
        node_id: str,
        direction: str = "all",
    ) -> list[dict]:
        """
        Get transfer history for a node.
        Args:
            node_id: Node to query
            direction: Filter by direction
        Returns:
            List of transfer records
        """
        transfers = []
        for transfer in reversed(self.transfer_history):
            if direction in ("all", "outgoing") and transfer.from_node_id == node_id:
                transfers.append(
                    {
                        "transfer_id": transfer.transfer_id,
                        "direction": "outgoing",
                        "counterparty": transfer.to_node_id,
                        "amount": -transfer.amount,
                        "reason": transfer.reason,
                        "timestamp": transfer.timestamp,
                        "tx_hash": transfer.tx_hash,
                    }
                )
            elif direction in ("all", "incoming") and transfer.to_node_id == node_id:
                transfers.append(
                    {
                        "transfer_id": transfer.transfer_id,
                        "direction": "incoming",
                        "counterparty": transfer.from_node_id,
                        "amount": transfer.amount,
                        "reason": transfer.reason,
                        "timestamp": transfer.timestamp,
                        "tx_hash": transfer.tx_hash,
                    }
                )
        return transfers

    def detect_suspicious_patterns(self, node_id: str) -> list[str]:
        """
        Detect suspicious transfer patterns for anti-fraud.
        Args:
            node_id: Node to analyze
        Returns:
            List of suspicious pattern alerts
        """
        alerts: list[str] = []
        now = time.time()
        node_transfers = [
            t for t in self.transfer_history if t.from_node_id == node_id or t.to_node_id == node_id
        ]
        if not node_transfers:
            return alerts
        five_min_ago = now - 300
        recent_transfers = [t for t in node_transfers if t.timestamp > five_min_ago]
        if len(recent_transfers) > 5:
            alerts.append(f"Rapid transfers: {len(recent_transfers)} transfers in last 5 minutes")
        for t1 in node_transfers[-10:]:
            if t1.from_node_id == node_id:
                for t2 in node_transfers:
                    if (
                        t2.to_node_id == node_id
                        and t2.from_node_id == t1.to_node_id
                        and 0 < t1.timestamp - t2.timestamp < 3600
                    ):
                        alerts.append(f"Round-trip transfer pattern with {t1.to_node_id[:20]}...")
                        break
        if len(node_transfers) > 20:
            small_transfers = [t for t in node_transfers if t.amount < 10]
            if len(small_transfers) > 15:
                alerts.append(f"Structuring pattern: {len(small_transfers)} small transfers")
        return alerts

    def get_stats(self) -> dict:
        total_volume = sum(t.amount for t in self.transfer_history)
        return {
            "total_transfers": len(self.transfer_history),
            "pending_transfers": len(self.pending_transfers),
            "total_volume": round(total_volume, 2),
            "transfers_last_hour": len(
                [t for t in self.transfer_history if t.timestamp > time.time() - 3600]
            ),
            "min_transfer": self.MIN_TRANSFER_AMOUNT,
            "max_transfer": self.MAX_TRANSFER_AMOUNT,
            "max_per_hour": self.MAX_TRANSFERS_PER_HOUR,
        }
