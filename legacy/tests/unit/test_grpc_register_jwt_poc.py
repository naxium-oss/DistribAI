"""gRPC registration with JWT or PoC when lockdown is on."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from services_python.grpc_service import GrpcServiceHandler
from worker.src.distribai_proto import distribai_pb2


@pytest.mark.asyncio
async def test_grpc_register_accepts_valid_jwt_when_lockdown(monkeypatch):
    monkeypatch.setenv("REGISTRATION_REQUIRE_POC", "true")

    node_service = MagicMock()
    node_service.version = "test"
    node_service.connected_nodes = {}
    node_service.verify_jwt.return_value = {"sub": "test-node-01", "kind": "node"}

    handler = GrpcServiceHandler(node_service)
    handler.db = MagicMock()
    out_queue: asyncio.Queue = asyncio.Queue()
    node_id_ref: dict = {"id": None}
    register = distribai_pb2.RegisterSession(
        node_id="test-node-01",
        jwt_token="existing-jwt",
        hardware_json="{}",
    )

    await handler._handle_register(register, out_queue, node_id_ref)

    assert node_id_ref["id"] == "test-node-01"
    msg = await asyncio.wait_for(out_queue.get(), timeout=1.0)
    assert msg.register_ack.session_token == "existing-jwt"
    node_service.verify_jwt.assert_called_once()


@pytest.mark.asyncio
async def test_grpc_register_accepts_poc_challenge_when_lockdown(monkeypatch):
    monkeypatch.setenv("REGISTRATION_REQUIRE_POC", "true")

    node_service = MagicMock()
    node_service.version = "test"
    node_service.connected_nodes = {}
    node_service.verify_jwt.return_value = None
    node_service.poc_challenge.verify_challenge.return_value = True
    node_service._issue_jwt.return_value = "fresh-jwt"

    handler = GrpcServiceHandler(node_service)
    handler.db = MagicMock()
    out_queue: asyncio.Queue = asyncio.Queue()
    node_id_ref: dict = {"id": None}
    register = distribai_pb2.RegisterSession(
        node_id="test-node-01",
        challenge_id="chal-1",
        nonce="nonce-1",
        hardware_json="{}",
    )

    await handler._handle_register(register, out_queue, node_id_ref)

    assert node_id_ref["id"] == "test-node-01"
    msg = await asyncio.wait_for(out_queue.get(), timeout=1.0)
    assert msg.register_ack.session_token == "fresh-jwt"
