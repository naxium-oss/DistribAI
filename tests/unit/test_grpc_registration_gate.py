"""gRPC open-registration gate mirrors HTTP /v1/nodes/register policy."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from services_python.grpc_service import GrpcServiceHandler
from worker.src.distribai_proto import distribai_pb2


@pytest.mark.asyncio
async def test_grpc_register_blocked_when_poc_required(monkeypatch):
    monkeypatch.setenv("REGISTRATION_REQUIRE_POC", "true")

    node_service = MagicMock()
    node_service.version = "test"
    node_service.connected_nodes = {}
    handler = GrpcServiceHandler(node_service)

    out_queue: asyncio.Queue = asyncio.Queue()
    node_id_ref: dict = {"id": None}
    register = distribai_pb2.RegisterSession(
        node_id="test-node-01",
        hardware_json="{}",
    )

    await handler._handle_register(register, out_queue, node_id_ref)

    assert node_id_ref["id"] is None
    assert "test-node-01" not in node_service.connected_nodes
    msg = await asyncio.wait_for(out_queue.get(), timeout=1.0)
    assert msg.register_ack.session_token == ""
