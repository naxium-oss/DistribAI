"""Integration smoke: create job via admin API and see it in the queue."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import tarfile
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from services_python.bundle_store import load_bundle
from services_python.db_manager import DBManager
from services_python.orchestrator_grpc import NodeService, _make_admin_app


def _minimal_script_b64() -> str:
    body = (
        b"import json\n"
        b'with open("results.json", "w") as f:\n'
        b'    json.dump({"ok": True}, f)\n'
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="run.py")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
async def admin_app_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_HOST", "127.0.0.1")
    monkeypatch.delenv("ADMIN_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("DISTRIBAI_BUNDLE_DIR", str(tmp_path / "bundles"))

    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "lifecycle.db"), str(schema_path))
    node_service = NodeService(db)
    app = _make_admin_app(node_service)
    try:
        yield app, node_service, db
    finally:
        await node_service.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_job_visible_in_admin_list(admin_app_jobs):
    app, _node_service, db = admin_app_jobs
    client = TestClient(TestServer(app))
    async with client:
        create = await client.post(
            "/admin/jobs",
            data=json.dumps(
                {
                    "steps": 2,
                    "batch_size": 4,
                    "job_type": "fine_tune",
                    "base_model": "uploaded-architecture",
                    "architecture_config": {
                        "family": "gru",
                        "dim": 128,
                        "gru_layers": 1,
                        "seq_len": 64,
                    },
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        assert create.status == 200
        created = await create.json()
        assert created.get("ok") is True
        job_id = created["job_id"]
        assert job_id

        listing = await client.get("/admin/jobs?active_only=false")
        assert listing.status == 200
        body = await listing.json()
        ids = {job["job_id"] for job in body.get("jobs", [])}
        assert job_id in ids
        assert "queue_fleet" in body
        assert "connected_count" in body["queue_fleet"]
        job_row = next(j for j in body["jobs"] if j["job_id"] == job_id)
        assert "queue_blockers" in job_row

        tasks = await asyncio.to_thread(db.get_queued_tasks)
        job_tasks = [task for task in tasks if task["job_id"] == job_id]
        assert job_tasks
        for task in job_tasks:
            task_hparams = json.loads(task["hparams_json"])
            assert task_hparams["architecture_config"]["family"] == "gru"
            assert task_hparams["architecture_config"]["version"] == 1
            assert task_hparams["architecture_config"]["architecture"] == "gru"

        detail = await client.get(f"/admin/jobs/{job_id}")
        assert detail.status == 200
        detail_body = await detail.json()
        assert "queue_blockers" in detail_body


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_v1_job_persists_normalized_architecture(admin_app_jobs):
    app, _node_service, db = admin_app_jobs
    client = TestClient(TestServer(app))
    async with client:
        create = await client.post(
            "/v1/jobs",
            data=json.dumps(
                {
                    "base_model": "uploaded-architecture",
                    "dataset_ref": "s3://bucket/data.json",
                    "steps": 2,
                    "architecture_config": {
                        "family": "gated_conv",
                        "dim": 128,
                        "n_logical_layers": 2,
                        "conv_kernel": 3,
                    },
                    "hyperparams": {"lr": 0.01},
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        assert create.status == 200
        created = await create.json()
        job_id = created["job_id"]

        tasks = await asyncio.to_thread(db.get_queued_tasks)
        job_tasks = [task for task in tasks if task["job_id"] == job_id]
        assert job_tasks
        for task in job_tasks:
            task_hparams = json.loads(task["hparams_json"])
            assert task_hparams["lr"] == 0.01
            assert task_hparams["architecture_config"] == {
                "version": 1,
                "family": "gated_conv",
                "architecture": "gated_conv",
                "dim": 128,
                "n_logical_layers": 2,
                "conv_kernel": 3,
            }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_script_job_persists_bundle(admin_app_jobs):
    app, _node_service, db = admin_app_jobs
    client = TestClient(TestServer(app))
    async with client:
        create = await client.post(
            "/admin/jobs",
            data=json.dumps(
                {
                    "steps": 2,
                    "batch_size": 4,
                    "script_package_b64": _minimal_script_b64(),
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        assert create.status == 200
        created = await create.json()
        task_id = created.get("task_id")
        assert task_id
        stored = load_bundle(task_id)
        assert stored is not None
        assert len(stored) > 20

        tasks = await asyncio.to_thread(db.get_queued_tasks)
        task_row = next(t for t in tasks if t["task_id"] == task_id)
        hparams = json.loads(task_row["hparams_json"])
        assert hparams.get("execution_paradigm") == "script"
