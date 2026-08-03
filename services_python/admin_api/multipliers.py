"""Operator endpoints for credit boost / surge controls."""

from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web

from services_python.credit_multipliers import CreditMultiplierEngine

logger = logging.getLogger(__name__)

# Surge windows are time-boxed; cap at 24h so a bad request can't create a
# near-permanent global multiplier.
MAX_SURGE_DURATION_SECONDS = 24 * 3600.0


class MultipliersHandler:
    """Inspect multiplier engine state and fire global surges."""

    def __init__(
        self,
        credit_multipliers: CreditMultiplierEngine,
        node_service: Any = None,
    ) -> None:
        self.credit_multipliers = credit_multipliers
        self.node_service = node_service

    def _require_admin(self, req: web.Request) -> None:
        """Handler-level admin gate (in addition to the auth middleware)."""
        if self.node_service is not None:
            self.node_service._authenticate_request(req, required_kind="admin")

    async def get_stats(self, req: web.Request) -> web.Response:
        """Live statistics from the credit multiplier engine."""
        self._require_admin(req)
        stats = self.credit_multipliers.get_stats()
        return web.json_response(stats)

    async def trigger_surge(self, req: web.Request) -> web.Response:
        """Begin a time-boxed global surge (default window: one hour).

        Malformed JSON is a client error (400), not a silent fallback — the
        previous behavior swallowed the parse failure and fired a full-length
        surge anyway.
        """
        self._require_admin(req)
        duration = 3600.0  # seconds when the body omits duration
        if req.can_read_body:
            try:
                data = await req.json()
            except json.JSONDecodeError:
                return web.json_response({"error": "invalid JSON"}, status=400)
            if data is not None and not isinstance(data, dict):
                return web.json_response({"error": "body must be a JSON object"}, status=400)
            if isinstance(data, dict) and "duration" in data:
                raw = data["duration"]
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    return web.json_response(
                        {"error": "duration must be a number of seconds"}, status=400
                    )
                duration = float(raw)
                if not 0 < duration <= MAX_SURGE_DURATION_SECONDS:
                    return web.json_response(
                        {
                            "error": "duration must be between 1 and "
                            f"{int(MAX_SURGE_DURATION_SECONDS)} seconds"
                        },
                        status=400,
                    )
        logger.info("Credit surge triggered for %.0f seconds", duration)
        self.credit_multipliers.trigger_surge(duration)
        return web.json_response(
            {
                "ok": True,
                "surge_triggered": True,
                "duration_seconds": duration,
                "surge_active": True,
                "surge_multiplier": self.credit_multipliers.SURGE_BOOST_MULTIPLIER,
            }
        )
