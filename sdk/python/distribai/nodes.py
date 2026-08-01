"""
Node management API for DistribAI
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from .client import DistribAIError

if TYPE_CHECKING:
    from .client import DistribAIClient


class NodeStatus(Enum):
    """
    Status of a compute node in DistribAI.

    Attributes:
        IDLE: Node is available for work
        WORKING: Node is currently processing a task
        DEGRADED: Node is degraded
        OFFLINE: Node is not connected
        BENCHMARKING: Node is benchmarking

    Example:
        status = NodeStatus.IDLE
        print(f"Node status: {status.value}")
    """

    IDLE = "idle"
    WORKING = "working"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    BENCHMARKING = "benchmarking"


@dataclass
class NodeHardware:
    gpu_model: str | None = None
    vram_gb: float | None = None
    compute_score: float | None = None
    driver_version: str | None = None
    os: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> NodeHardware:
        return cls(
            gpu_model=data.get("gpu_model"),
            vram_gb=data.get("vram_gb"),
            compute_score=data.get("compute_score"),
            driver_version=data.get("driver_version"),
            os=data.get("os"),
        )


@dataclass
class Node:
    """
    Represents a worker node in DistribAI.
    Attributes:
        id: Unique node ID
        status: Current node status
        hardware: Hardware information
        reliability_score: Reliability score (0-1)
        credits_earned: Total credits earned
        current_job_id: Currently assigned job (if any)
        last_heartbeat: Last heartbeat timestamp
        registered_at: Registration timestamp
        region: Geographic region
    """

    id: str
    status: NodeStatus
    hardware: NodeHardware
    reliability_score: float = 1.0
    credits_earned: float = 0.0
    current_job_id: str | None = None
    last_heartbeat: str | None = None
    registered_at: str | None = None
    region: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> Node:
        return cls(
            id=data["node_id"],
            status=NodeStatus(data.get("status", "offline")),
            hardware=NodeHardware.from_dict(data.get("hardware", {})),
            reliability_score=data.get("reliability_score", 1.0),
            credits_earned=data.get("credits_earned", 0.0),
            current_job_id=data.get("current_job_id"),
            last_heartbeat=data.get("last_heartbeat"),
            registered_at=data.get("registered_at"),
            region=data.get("region"),
        )

    @property
    def is_online(self) -> bool:
        return self.status != NodeStatus.OFFLINE

    @property
    def is_working(self) -> bool:
        return self.status == NodeStatus.WORKING


class NodesAPI:
    def __init__(self, client: DistribAIClient):
        self._client = client

    async def register(
        self,
        invite_code: str | None = None,
        public_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Register a new node.
        Args:
            invite_code: Optional invite code
            public_key: Optional public key for authentication
        Returns:
            Registration response with node_id and jwt_token
        """
        data = {
            "invite_code": invite_code,
            "public_key": public_key,
        }
        response = await self._client._request("POST", "/v1/nodes/register", json=data)
        return response

    async def list(
        self,
        status: NodeStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Node]:
        """
        List nodes in the network.
        Args:
            status: Filter by status
            limit: Maximum nodes to return
            offset: Pagination offset
        Returns:
            List of Node instances
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value
        response = await self._client._request("GET", "/admin/nodes", params=params)
        nodes = response.get("nodes", [])
        return [Node.from_dict(n) for n in nodes]

    async def get(self, node_id: str) -> Node:
        """
        Get specific node information.
        Args:
            node_id: Node ID
        Returns:
            Node instance
        """
        nodes = await self.list()
        for node in nodes:
            if node.id == node_id:
                return node
        raise DistribAIError(f"node not found: {node_id}")

    async def get_stats(self) -> dict[str, Any]:
        """
        Get network-wide node statistics.
        Returns:
            Dict with total nodes, online count, compute capacity, etc.
        """
        nodes = await self.list()
        return {
            "total_nodes": len(nodes),
            "online_nodes": sum(1 for node in nodes if node.is_online),
            "working_nodes": sum(1 for node in nodes if node.is_working),
        }

    async def update_heartbeat(
        self,
        node_id: str,
        seq: int,
        vram_free_mb: int,
        gpu_util_pct: float,
        current_task_id: str | None = None,
    ) -> bool:
        """
        Send heartbeat update for a node.
        Args:
            node_id: Node ID
            seq: Heartbeat sequence number
            vram_free_mb: Free VRAM in MB
            gpu_util_pct: GPU utilization percentage
            current_task_id: Currently executing task (if any)
        Returns:
            True if heartbeat accepted
        """
        data = {
            "node_id": node_id,
            "seq": seq,
            "vram_free_mb": vram_free_mb,
            "gpu_util_pct": gpu_util_pct,
            "current_task_id": current_task_id,
        }
        await self._client._request("POST", "/v1/nodes/heartbeat", json=data)
        return True
