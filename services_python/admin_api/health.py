"""Fleet liveness payload served on the orchestrator admin HTTP listener."""

import asyncio

from aiohttp import web

from services_python.db_manager import DBManager

# Forward type name only; NodeService is injected at construction time.
NodeService = None


class HealthHandler:
    """JSON health snapshot for operator dashboards and uptime probes."""

    def __init__(self, db: DBManager, node_service: "NodeService") -> None:
        self.db = db
        self.node_service = node_service

    async def get(self, req: web.Request) -> web.Response:
        """Emit ok, queue depth, connected workers, and job-submit gate."""
        # Circular import: flag lives on the orchestrator entry module.
        from services_python.orchestrator_grpc import JOB_SUBMISSION_AVAILABLE

        return web.json_response(
            {
                "ok": True,
                "timestamp": asyncio.get_event_loop().time(),
                "active_nodes": len(self.node_service.connected_nodes),
                "queued_jobs": await asyncio.to_thread(self.db.get_queue_depth),
                "running_jobs": len(self.node_service.pending_assignments),
                "job_submission_available": JOB_SUBMISSION_AVAILABLE,
            }
        )
