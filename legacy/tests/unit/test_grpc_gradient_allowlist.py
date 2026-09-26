"""Unit tests for orchestrator gradient URL allowlist enforcement."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services_python.grpc_service import GrpcServiceHandler


@pytest.mark.asyncio
async def test_load_gradient_payload_blocks_remote_https(monkeypatch):
    monkeypatch.setenv("ALLOWED_BLOB_HOSTS", "127.0.0.1,localhost")

    handler = GrpcServiceHandler(MagicMock())
    result = await handler._load_gradient_payload("https://evil.example/grad.json")
    assert result is None
