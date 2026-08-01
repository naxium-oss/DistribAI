"""Registered-worker listing, contribute toggles, disconnect, and sync."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from aiohttp import web

from services_python.db_manager import DBManager
from services_python.pagination import PaginationHeaders, paginate_list, parse_pagination_params

if TYPE_CHECKING:
    from services_python.orchestrator_grpc import NodeService


class NodesHandler:
    """Admin views over SQLite registrations and live gRPC sessions."""

    def __init__(self, db: DBManager, node_service: NodeService) -> None:
        self.db = db
        self.node_service = node_service

    async def list(self, req: web.Request) -> web.Response:
        """All nodes with online flag, bench score, and credit balance."""
        self.node_service._authenticate_request(req, required_kind="admin")
        nodes = await asyncio.to_thread(self.db.get_all_nodes)
        credits_map = await asyncio.to_thread(self.db.list_all_credits)
        connected = set(self.node_service.connected_nodes.keys())
        for node in nodes:
            node_id = str(node.get("node_id") or "")
            node["online"] = node_id in connected
            bench = node.get("benchmark") if isinstance(node.get("benchmark"), dict) else {}
            node["benchmark_score"] = bench.get("overall_score") if isinstance(bench, dict) else None
            node["current_job"] = node.get("current_task_id")
            hw = node.get("hardware") if isinstance(node.get("hardware"), dict) else {}
            node["ip"] = hw.get("ip") or hw.get("public_ip") or ""
            node["hardware_summary"] = hw.get("summary") or hw.get("gpu_name") or ""
            node["credits"] = float((credits_map.get(node_id) or {}).get("balance") or 0)
        return web.json_response({"nodes": nodes})

    async def set_contributing(self, req: web.Request) -> web.Response:
        """Set the contributing flag persisted for a node."""
        self.node_service._authenticate_request(req, required_kind="admin")
        node_id = req.match_info.get("node_id")
        if not node_id:
            return web.json_response({"error": "missing node_id"}, status=400)

        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        contributing = bool(body.get("contributing", True))
        await asyncio.to_thread(self.db.set_node_contributing, node_id, contributing)

        return web.json_response({"ok": True, "node_id": node_id, "contributing": contributing})

    async def disconnect(self, req: web.Request) -> web.Response:
        """Force-close a worker session and mark it non-contributing."""
        self.node_service._authenticate_request(req, required_kind="admin")
        node_id = req.match_info.get("node_id")
        if not node_id:
            return web.json_response({"error": "missing node_id"}, status=400)

        was_connected = False
        queue = self.node_service.connected_nodes.pop(node_id, None)
        if queue is not None:
            was_connected = True
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self.node_service.pending_assignments.pop(node_id, None)
        await asyncio.to_thread(self.db.set_node_contributing, node_id, False)
        await asyncio.to_thread(
            self.db.update_heartbeat, node_id, "disconnected", int(time.time())
        )
        return web.json_response(
            {"ok": True, "node_id": node_id, "was_connected": was_connected}
        )

    async def sync_all(self, req: web.Request) -> web.Response:
        """Recompute online markers against the live connection map."""
        self.node_service._authenticate_request(req, required_kind="admin")
        nodes = await asyncio.to_thread(self.db.get_all_nodes)
        connected = set(self.node_service.connected_nodes.keys())
        for node in nodes:
            node["online"] = node.get("node_id") in connected
        return web.json_response(
            {
                "ok": True,
                "synced": len(nodes),
                "connected": len(connected),
                "nodes": nodes,
            }
        )

    async def list_paginated(self, req: web.Request) -> web.Response:
        """Sorted, paged node list for large operator fleets."""
        self.node_service._authenticate_request(req, required_kind="admin")
        allowed = {
            "node_id",
            "status",
            "last_heartbeat_ts",
            "created_ts",
            "updated_ts",
            "reliability_score",
            "jobs_completed",
        }
        params = parse_pagination_params(dict(req.query), allowed_sort_columns=allowed)

        nodes = await asyncio.to_thread(self.db.get_all_nodes)
        reverse = params.sort_order == "desc"

        try:
            nodes = sorted(nodes, key=lambda x: x.get(params.sort_by, 0), reverse=reverse)
        except (TypeError, KeyError):
            nodes = sorted(
                nodes,
                key=lambda x: str(x.get("node_id", "")),
                reverse=reverse,
            )

        result = paginate_list(nodes, params)
        headers = PaginationHeaders.build(
            total=result.pagination["total"], page=params.page, per_page=params.per_page
        )

        return web.json_response(result.to_dict(), headers=headers)
