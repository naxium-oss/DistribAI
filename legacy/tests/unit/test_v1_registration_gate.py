"""Tests for v1 open-registration gate."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp.test_utils import make_mocked_request

from services_python.admin_api.v1 import V1Handler


@pytest.mark.asyncio
async def test_register_node_blocked_when_poc_required(monkeypatch):
    monkeypatch.setenv("ADMIN_REQUIRE_AUTH", "true")
    monkeypatch.setenv("REGISTRATION_REQUIRE_POC", "true")

    handler = V1Handler(MagicMock(), MagicMock())
    body = {"node_id": "test-node-01", "os": "linux", "gpu_model": "cpu", "vram_mb": 0}
    req = make_mocked_request(
        "POST",
        "/v1/nodes/register",
        payload=b'{"node_id":"test-node-01"}',
        headers={"Content-Type": "application/json"},
    )
    req.json = AsyncMock(return_value=body)

    resp = await handler.register_node(req)
    assert resp.status == 403
    payload = json.loads(resp.text)
    assert payload["error"] == "registration_requires_poc"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, expected_error",
    [
        ({}, "overall_score is required"),
        ([], "benchmark payload must be an object"),
        ({"overall_score": float("nan")}, "finite and between 0 and 100"),
        ({"overall_score": 50, "metric": float("inf")}, "JSON-finite values"),
    ],
)
async def test_submit_benchmark_rejects_invalid_payloads(body, expected_error):
    db = MagicMock()
    node_service = MagicMock()
    node_service._authenticate_request.return_value = {"sub": "node-1"}
    handler = V1Handler(db, node_service)
    req = make_mocked_request("POST", "/v1/nodes/benchmark")
    req.json = AsyncMock(return_value=body)

    resp = await handler.submit_benchmark(req)

    assert resp.status == 400
    assert expected_error in json.loads(resp.text)["error"]
    db.update_node_benchmark.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_hparams", ["not-an-object", ["bad"], 7])
async def test_create_job_rejects_non_object_hparams(bad_hparams):
    db = MagicMock()
    node_service = MagicMock()
    handler = V1Handler(db, node_service)
    req = make_mocked_request("POST", "/v1/jobs")
    req.json = AsyncMock(
        return_value={"base_model": "uploaded-architecture", "steps": 2, "hparams": bad_hparams}
    )

    response = await handler.create_job(req)

    assert response.status == 400
    assert json.loads(response.text)["error"] == "hparams must be an object"
    db.create_job.assert_not_called()


@pytest.mark.asyncio
async def test_create_job_persists_normalized_architecture_config():
    db = MagicMock()
    db.create_job.return_value = "job-v1-architecture"
    node_service = MagicMock()
    handler = V1Handler(db, node_service)
    req = make_mocked_request("POST", "/v1/jobs")
    req.json = AsyncMock(
        return_value={
            "base_model": "uploaded-architecture",
            "dataset_ref": "s3://bucket/data.json",
            "steps": 2,
            "architecture_config": {"family": "gru", "dim": 128, "gru_layers": 1},
            "hyperparams": {"lr": 0.01},
        }
    )

    response = await handler.create_job(req)

    assert response.status == 200
    assert json.loads(response.text) == {"ok": True, "job_id": "job-v1-architecture"}
    db.create_job.assert_called_once()
    hyperparams = db.create_job.call_args.kwargs["hyperparams"]
    assert hyperparams["lr"] == 0.01
    assert hyperparams["architecture_config"] == {
        "version": 1,
        "family": "gru",
        "architecture": "gru",
        "dim": 128,
        "gru_layers": 1,
    }
