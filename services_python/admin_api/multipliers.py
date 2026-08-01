"""Operator endpoints for credit boost / surge controls."""

from aiohttp import web

from services_python.credit_multipliers import CreditMultiplierEngine


class MultipliersHandler:
    """Inspect multiplier engine state and fire global surges."""

    def __init__(self, credit_multipliers: CreditMultiplierEngine) -> None:
        self.credit_multipliers = credit_multipliers

    async def get_stats(self, req: web.Request) -> web.Response:
        """Live statistics from the credit multiplier engine."""
        stats = self.credit_multipliers.get_stats()
        return web.json_response(stats)

    async def trigger_surge(self, req: web.Request) -> web.Response:
        """Begin a time-boxed global surge (default window: one hour)."""
        duration = 3600  # seconds if the body omits duration
        try:
            data = await req.json()
            if data is not None and 'duration' in data:
                duration = float(data['duration'])
        except Exception:
            pass
        self.credit_multipliers.trigger_surge(duration)
        return web.json_response({
            "ok": True,
            "surge_triggered": True,
            "duration_seconds": duration,
            "surge_active": True,
            "surge_multiplier": self.credit_multipliers.SURGE_BOOST_MULTIPLIER
        })
