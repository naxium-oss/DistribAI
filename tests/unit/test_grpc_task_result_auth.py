"""gRPC task result must match registered session node."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services_python.grpc_service import GrpcServiceHandler
from worker.src.distribai_proto import distribai_pb2


@pytest.fixture
def handler():
    node_service = MagicMock()
    node_service.log_lines = []
    node_service.pending_assignments = {}
    node_service._safe_json = lambda raw: __import__("json").loads(raw) if raw else {}
    node_service.db = MagicMock()
    return GrpcServiceHandler(node_service)


@pytest.mark.asyncio
async def test_handle_result_ignored_before_register(handler):
    result = distribai_pb2.TaskResult(
        task_id="t1",
        job_id="j1",
        node_id="evil-node",
        status="success",
        output_json='{"credits_earned": 9999}',
    )
    await handler._handle_result(result, None)
    handler.node_service.db.update_task_result.assert_not_called()
    handler.node_service.record_credit_earn.assert_not_called()


@pytest.mark.asyncio
async def test_handle_result_rejects_node_id_mismatch(handler):
    result = distribai_pb2.TaskResult(
        task_id="t1",
        job_id="j1",
        node_id="other-node",
        status="success",
        output_json="{}",
    )
    await handler._handle_result(result, "session-node")
    handler.node_service.db.update_task_result.assert_not_called()


@pytest.mark.asyncio
async def test_handle_progress_rejects_node_id_mismatch(handler):
    progress = distribai_pb2.TaskProgress(
        node_id="other-node",
        job_id="j1",
        task_id="t1",
        step=1,
        loss=0.5,
        ts=1,
    )
    await handler._handle_progress(progress, "session-node")
    handler.node_service.db.update_task_progress.assert_not_called()


@pytest.mark.asyncio
async def test_handle_result_caps_reported_credits(handler):
    result = distribai_pb2.TaskResult(
        task_id="t1",
        job_id="j1",
        node_id="session-node",
        status="success",
        output_json='{"credits_earned": 999999}',
    )
    await handler._handle_result(result, "session-node")
    handler.node_service.record_credit_earn.assert_called_once()
    _args = handler.node_service.record_credit_earn.call_args[0]
    assert _args[0] == "session-node"
    assert _args[1] == 100.0
