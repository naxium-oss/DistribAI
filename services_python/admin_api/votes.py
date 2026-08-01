"""Priority-vote create/list/cast handlers for admin and v1 clients."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from aiohttp import web

from services_python.db_manager import DBManager
from services_python.schemas import validate_vote
from worker.src.daemon.voting_system import VoteType, VotingSystem

if TYPE_CHECKING:
    from services_python.orchestrator_grpc import NodeService


class VotesHandler:
    """Governance votes that reshape queue priority via pledged credits."""

    def __init__(
        self,
        db: DBManager,
        voting_system: VotingSystem,
        node_service: NodeService,
    ) -> None:
        self.db = db
        self.voting_system = voting_system
        self.node_service = node_service

    async def list(self, req: web.Request) -> web.Response:
        """Open voting rounds currently tracked in memory."""
        self.node_service._authenticate_request(req, required_kind="admin")
        active = self.voting_system.get_active_votes()
        return web.json_response({"votes": active, "count": len(active)})

    async def get(self, req: web.Request) -> web.Response:
        """Status payload for a single ``vote_id``."""
        self.node_service._authenticate_request(req, required_kind="admin")
        vote_id = req.match_info.get("vote_id")
        if not vote_id:
            return web.json_response({"error": "missing vote_id"}, status=400)

        status = self.voting_system.get_vote_status(vote_id)
        if not status:
            return web.json_response({"error": "not found"}, status=404)

        return web.json_response(status)

    async def create(self, req: web.Request) -> web.Response:
        """Validate body and open a fresh voting round."""
        self.node_service._authenticate_request(req, required_kind="admin")
        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        valid, error, _validated = validate_vote(body)
        if not valid:
            return web.json_response({"error": error}, status=400)

        try:
            vote_id = self.voting_system.create_vote(
                proposer=body["proposer"],
                vote_type=VoteType(body.get("vote_type", "job_priority")),
                title=body["title"],
                description=body.get("description", ""),
                options=body.get("options", ["yes", "no"]),
            )
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        return web.json_response({"ok": True, "vote_id": vote_id})

    async def cast(self, req: web.Request) -> web.Response:
        """Record a voter's chosen option on an open round."""
        self.node_service._authenticate_request(req, required_kind="admin")
        vote_id = req.match_info.get("vote_id")
        if not vote_id:
            return web.json_response({"error": "missing vote_id"}, status=400)

        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        voter_id = body.get("voter_id")
        option = body.get("option")

        if not voter_id or not option:
            return web.json_response({"error": "missing voter_id or option"}, status=400)

        success = self.voting_system.cast_vote(voter_id, vote_id, option)
        if not success:
            return web.json_response({"error": "vote failed"}, status=400)

        return web.json_response({"ok": True})

    async def vote_v1(self, req: web.Request) -> web.Response:
        """v1 route: spend credits to boost a job's priority score."""
        claims = self.node_service._authenticate_request(req, required_kind="node")
        node_id = claims["sub"]

        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        job_id = body.get("job_id")
        credits = body.get("credits", 0)

        if not job_id or credits <= 0:
            return web.json_response({"error": "invalid parameters"}, status=400)

        if not self.voting_system.spend_credits(node_id, credits, f"Vote on job {job_id}"):
            return web.json_response({"error": "insufficient credits"}, status=400)

        await asyncio.to_thread(self.db.record_vote, job_id, node_id, int(credits))
        job = await asyncio.to_thread(self.db.get_job, job_id)
        if not job:
            return web.json_response({"error": "job not found"}, status=404)

        priority_score = float(job["total_votes"]) * float(job.get("vote_weight", 1.0))

        return web.json_response(
            {
                "ok": True,
                "job_id": job_id,
                "credits_spent": credits,
                "new_priority_score": priority_score,
            }
        )

    async def list_v1(self, req: web.Request) -> web.Response:
        """v1 route: enumerate active votes for node clients."""
        active = self.voting_system.get_active_votes()
        return web.json_response({"votes": active})
