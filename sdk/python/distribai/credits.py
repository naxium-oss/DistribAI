"""
Credit management API for DistribAI
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import DistribAIClient


@dataclass
class CreditBalance:
    """
    Credit balance information.

    Attributes:
        confirmed: Confirmed available credits
        pending: Pending credits (not yet confirmed)
        lifetime_earned: Total credits earned over lifetime
        lifetime_votes_cast: Total votes cast
    """

    confirmed: float
    pending: float = 0.0
    lifetime_earned: float = 0.0
    lifetime_votes_cast: int = 0


@dataclass
class Transaction:
    """
    Credit transaction record.
    Attributes:
        id: Transaction ID
        type: Transaction type (earn, spend, transfer)
        amount: Credit amount (positive or negative)
        balance_after: Balance after this transaction
        description: Transaction description
        timestamp: Transaction timestamp
        ref_id: Reference ID (job_id, vote_id, etc.)
    """

    id: str
    type: str
    amount: float
    balance_after: float
    description: str
    timestamp: str
    ref_id: str | None = None


class CreditsAPI:
    def __init__(self, client: DistribAIClient):
        self._client = client

    async def balance(self) -> CreditBalance:
        """
        Get current credit balance.
        Returns:
            CreditBalance instance
        Example:
            >>> balance = await client.credits.balance()
            >>> print(f"Available: {balance.confirmed}")
            >>> print(f"Lifetime earned: {balance.lifetime_earned}")
        """
        response = await self._client._request("GET", "/v1/credits/balance")
        return CreditBalance(
            confirmed=response.get("confirmed", 0.0),
            pending=response.get("pending", 0.0),
            lifetime_earned=response.get("lifetime_earned", 0.0),
            lifetime_votes_cast=response.get("lifetime_votes_cast", 0),
        )

    async def history(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Transaction]:
        """
        Get credit transaction history.
        Args:
            limit: Maximum transactions to return
            offset: Pagination offset
        Returns:
            List of Transaction records
        """
        response = await self._client._request(
            "GET", "/v1/credits/transfers", params={"limit": limit, "offset": offset}
        )
        transactions = response.get("transactions", [])
        return [
            Transaction(
                id=t["id"],
                type=t["type"],
                amount=t["amount"],
                balance_after=t["balance_after"],
                description=t.get("description", ""),
                timestamp=t["timestamp"],
                ref_id=t.get("ref_id"),
            )
            for t in transactions
        ]

    async def transfer(
        self,
        to_node_id: str,
        amount: float,
        reason: str | None = None,
    ) -> bool:
        """
        Transfer credits to another node.
        Args:
            to_node_id: Destination node ID
            amount: Amount to transfer
            reason: Optional transfer reason
        Returns:
            True if transfer successful
        Raises:
            InsufficientCreditsError: If balance is insufficient
        """
        data = {
            "to_node_id": to_node_id,
            "amount": amount,
            "reason": reason,
        }
        await self._client._request("POST", "/v1/credits/transfer", json=data)
        return True

    async def estimated_earning_rate(self) -> dict:
        """
        Get estimated credit earning rate based on current hardware.
        Returns:
            Dict with hourly rate and multipliers
        """
        response = await self._client._request("GET", "/v1/credits/multipliers")
        return {
            "credits_per_hour": response.get("credits_per_hour", 0.0),
            "base_rate": response.get("base_rate", 0.0),
            "multipliers": response.get("multipliers", {}),
            "hardware_score": response.get("hardware_score", 0.0),
        }
