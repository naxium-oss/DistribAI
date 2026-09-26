"""Admin/node HTTP handlers for balances, transfers, and surge opt-in."""

import asyncio
import json
import logging
from datetime import UTC, datetime

from aiohttp import web

from services_python.credit_multipliers import CreditMultiplierEngine
from services_python.credit_transfers import CreditTransferManager
from services_python.db_manager import DBManager
from services_python.pagination import PaginationHeaders, paginate_list, parse_pagination_params
from worker.src.daemon.credit_ledger import CreditLedger

logger = logging.getLogger(__name__)

# Injected at runtime; avoids importing orchestrator_grpc at module load.
NodeService = None


class CreditsHandler:
    """Balance maps, transfers, and surge controls for admin and v1 clients."""

    def __init__(
        self,
        db: DBManager,
        credit_ledger: CreditLedger,
        credit_transfers: CreditTransferManager,
        node_service: "NodeService",
    ) -> None:
        self.db = db
        self.credit_ledger = credit_ledger
        self.credit_transfers = credit_transfers
        self.node_service = node_service

    async def list(self, req: web.Request) -> web.Response:
        """Full credit map with lifetime-issued total and node count."""
        self.node_service._authenticate_request(req, required_kind="admin")
        credits = await asyncio.to_thread(self.db.list_all_credits)
        total_issued = sum(float(info.get("lifetime") or 0) for info in credits.values())
        return web.json_response(
            {
                "credits": credits,
                "total_issued": total_issued,
                "node_count": len(credits),
            }
        )

    async def get(self, req: web.Request) -> web.Response:
        """Single-node credit row shaped for operator dashboards."""
        self.node_service._authenticate_request(req, required_kind="admin")
        node_id = req.match_info.get("node_id")
        if not node_id:
            return web.json_response({"error": "missing node_id"}, status=400)

        credits = await asyncio.to_thread(self.db.get_node_credits, node_id)
        balance = float((credits or {}).get("balance") or 0)
        lifetime = float((credits or {}).get("lifetime") or 0)
        votes_cast = float((credits or {}).get("votes_cast") or 0)

        transactions: list[dict] = []
        history: list[dict] = []
        multipliers: list[dict] = []
        data_warnings: list[str] = []

        try:
            ledger_rows = self.credit_ledger.get_credit_history(node_id)
        except Exception:
            # A dashboard with partial data beats a 500, but the failure must
            # be visible to operators instead of silently rendering as empty.
            logger.exception("Credit ledger history lookup failed for node %s", node_id)
            ledger_rows = []
            data_warnings.append("ledger history unavailable")
        if isinstance(ledger_rows, list):
            daily: dict[str, float] = {}
            for row in ledger_rows:
                if not isinstance(row, dict):
                    continue
                amount = float(row.get("amount") or 0)
                ts = row.get("timestamp")
                job_id = row.get("job_id") or ""
                transactions.append(
                    {
                        "description": f"Ledger · {job_id}" if job_id else "Ledger entry",
                        "amount": amount,
                        "timestamp": ts,
                        "job_id": job_id or None,
                        "index": row.get("index"),
                        "source": "ledger",
                    }
                )
                if ts is not None:
                    try:
                        day = datetime.fromtimestamp(float(ts), tz=UTC).strftime("%Y-%m-%d")
                    except (TypeError, ValueError, OSError):
                        day = "unknown"
                    daily[day] = daily.get(day, 0.0) + max(0.0, amount)
            history = [{"date": day, "earned": round(val, 4)} for day, val in sorted(daily.items())]

        try:
            transfer_rows = self.credit_transfers.get_transfer_history(node_id)
        except Exception:
            logger.exception("Transfer history lookup failed for node %s", node_id)
            transfer_rows = []
            data_warnings.append("transfer history unavailable")
        if isinstance(transfer_rows, list):
            for row in transfer_rows:
                if not isinstance(row, dict):
                    continue
                amount = float(row.get("amount") or 0)
                direction = row.get("direction") or "transfer"
                counterparty = row.get("counterparty") or "peer"
                transactions.append(
                    {
                        "description": f"{str(direction).title()} · {counterparty}",
                        "amount": amount,
                        "timestamp": row.get("timestamp"),
                        "transfer_id": row.get("transfer_id"),
                        "source": "transfer",
                    }
                )

        transactions.sort(key=lambda t: float(t.get("timestamp") or 0), reverse=True)

        try:
            engine = getattr(self.node_service, "credit_multipliers", None)
            if engine is not None and hasattr(engine, "get_node_summary"):
                summary = engine.get_node_summary(node_id)
                if isinstance(summary, dict):
                    mult_map = summary.get("multipliers") or {}
                    labels = {
                        "base": "Base rate",
                        "surge_opt_in": "Community surge opt-in",
                        "reliability": "Reliability bonus",
                        "early_adopter": "Early adopter",
                        "error_penalty": "Error-rate penalty",
                        "low_demand": "Low network demand",
                        "surge_boost": "Surge boost",
                    }
                    if isinstance(mult_map, dict):
                        for key, label in labels.items():
                            val = mult_map.get(key)
                            if val is None:
                                continue
                            try:
                                multipliers.append(
                                    {
                                        "name": label,
                                        "key": key,
                                        "value": float(val),
                                        "active": key == "base" or float(val) != 1.0,
                                    }
                                )
                            except (TypeError, ValueError):
                                continue
                    final_val = summary.get("effective_multiplier")
                    if final_val is None and isinstance(mult_map, dict):
                        final_val = mult_map.get("final")
                    multipliers.append(
                        {
                            "name": "Effective total",
                            "key": "final",
                            "value": float(final_val or 1.0),
                            "active": True,
                        }
                    )
        except Exception:
            logger.exception("Multiplier summary lookup failed for node %s", node_id)
            multipliers = []
            data_warnings.append("multiplier summary unavailable")

        payload = {
            "node_id": node_id,
            "balance": balance,
            "confirmed": balance,
            "pending": 0.0,
            "lifetime": lifetime,
            "lifetime_earned": lifetime,
            "lifetime_votes_cast": votes_cast,
            "votes_cast": votes_cast,
            "history": history,
            "transactions": transactions,
            "multipliers": multipliers,
        }
        if data_warnings:
            payload["data_warnings"] = data_warnings
        return web.json_response(payload)

    async def list_paginated(self, req: web.Request) -> web.Response:
        """Page through credit rows when the fleet map is large."""
        self.node_service._authenticate_request(req, required_kind="admin")
        allowed = {"node_id", "balance", "lifetime", "votes_cast", "created_ts"}
        params = parse_pagination_params(dict(req.query), allowed_sort_columns=allowed)

        credits = await asyncio.to_thread(self.db.list_all_credits)
        credit_list = [{"node_id": k, **v} for k, v in credits.items()]
        result = paginate_list(credit_list, params)

        headers = PaginationHeaders.build(
            total=result.pagination["total"], page=params.page, per_page=params.per_page
        )

        return web.json_response(result.to_dict(), headers=headers)

    async def get_transfer_stats(self, req: web.Request) -> web.Response:
        """Aggregate counters from the transfer manager."""
        self.node_service._authenticate_request(req, required_kind="admin")
        stats = self.credit_transfers.get_stats()
        return web.json_response(stats)

    async def get_balance_v1(self, req: web.Request) -> web.Response:
        """v1 route: confirmed/pending/lifetime for the JWT subject."""
        claims = self.node_service._authenticate_request(req, required_kind="node")
        node_id = claims["sub"]

        row = await asyncio.to_thread(self.db.get_node_credits, node_id)
        confirmed = float(row["balance"]) if row else 0.0
        lifetime = float(row["lifetime"]) if row else 0.0
        return web.json_response(
            {
                "node_id": node_id,
                "confirmed": confirmed,
                "pending": 0.0,
                "lifetime_earned": lifetime,
            }
        )

    async def transfer_v1(self, req: web.Request) -> web.Response:
        """v1 route: peer-to-peer credit transfer from the caller."""
        claims = self.node_service._authenticate_request(req, required_kind="node")
        node_id = claims["sub"]

        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        recipient = body.get("recipient")
        amount = body.get("amount")
        memo = body.get("memo", "")

        if not recipient or not isinstance(amount, (int, float)) or amount <= 0:
            return web.json_response({"error": "invalid parameters"}, status=400)

        result = await self.credit_transfers.transfer(node_id, recipient, amount, memo)
        return web.json_response(result)

    async def get_transfer_history_v1(self, req: web.Request) -> web.Response:
        """v1 route: transfer history for the authenticated node."""
        claims = self.node_service._authenticate_request(req, required_kind="node")
        node_id = claims["sub"]

        history = self.credit_transfers.get_transfer_history(node_id)
        return web.json_response({"transfers": history})

    async def get_multiplier_status_v1(self, req: web.Request) -> web.Response:
        """v1 route: multiplier / surge summary for the caller."""
        claims = self.node_service._authenticate_request(req, required_kind="node")
        node_id = claims["sub"]

        multipliers: CreditMultiplierEngine = self.node_service.credit_multipliers

        summary = multipliers.get_node_summary(node_id)
        return web.json_response(summary)

    async def set_surge_opt_in_v1(self, req: web.Request) -> web.Response:
        """v1 route: enable or disable surge opt-in for the caller."""
        claims = self.node_service._authenticate_request(req, required_kind="node")
        node_id = claims["sub"]

        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        opt_in = bool(body.get("opt_in", False))
        self.node_service.credit_multipliers.set_surge_opt_in(node_id, opt_in)

        return web.json_response(
            {
                "node_id": node_id,
                "surge_opt_in": opt_in,
                "message": f"Surge opt-in {'enabled' if opt_in else 'disabled'}",
            }
        )
