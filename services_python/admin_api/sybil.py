"""HTTP surface for Sybil-risk aggregates and per-node reports."""

from aiohttp import web

from services_python.sybil_detector import SybilDetector


class SybilHandler:
    """Read-only Sybil detector views for operators."""

    def __init__(self, sybil_detector: SybilDetector) -> None:
        self.sybil_detector = sybil_detector

    async def get_stats(self, req: web.Request) -> web.Response:
        """Fleet-wide Sybil detector counters."""
        stats = self.sybil_detector.get_network_stats()
        return web.json_response(stats)

    async def get_node_report(self, req: web.Request) -> web.Response:
        """Risk report keyed by ``node_id`` path parameter."""
        node_id = req.match_info.get("node_id")
        if not node_id:
            return web.json_response({"error": "missing node_id"}, status=400)

        report = self.sybil_detector.get_account_report(node_id)
        return web.json_response(report)
