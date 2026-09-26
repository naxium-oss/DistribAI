"""Unit tests for job preflight and failure code attachment."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from services_python.db_manager import DBManager
from services_python.job_failure_codes import FAILURE_CODES, attach_failure_fields
from services_python.preflight import validate_script_tarball


def _tar_with_run_py() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"print('ok')\n"
        info = tarfile.TarInfo(name="run.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_preflight_accepts_minimal_run_py():
    ok, err, meta = validate_script_tarball(_tar_with_run_py())
    assert ok is True
    assert err is None
    assert meta["member_count"] == 1


def test_preflight_rejects_forbidden_env():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in (("run.py", b"x=1\n"), (".env", b"SECRET=1\n")):
            data = content
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    ok, err, _ = validate_script_tarball(buf.getvalue())
    assert ok is False
    assert "forbidden" in (err or "")


def test_attach_failure_fields_cancelled():
    row = attach_failure_fields({"status": "cancelled", "latest_reason": "job cancelled"})
    assert row["failure_code"] == "job_cancelled"
    assert "job_cancelled" in row["failure_code_catalog"]


def test_failure_codes_catalog_nonempty():
    assert "hash_mismatch" in FAILURE_CODES


def test_operator_retry_requeues_failed_task_and_clears_completed_ts(tmp_path):
    schema_path = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "retry.db"), str(schema_path))
    job_id = db.create_job(
        job_type="fine_tune",
        base_model="toy",
        dataset_ref="local",
        hyperparams={},
        total_steps=2,
        max_attempts=0,
    )
    task_id = db.get_queued_tasks()[0]["task_id"]
    db.update_task_result(
        task_id=task_id,
        node_id=None,
        status="failed",
        output_json=json.dumps({"error": "boom"}),
        gradient_blob_url="",
    )
    terminal_job = db.get_job(job_id)
    assert terminal_job is not None
    assert terminal_job["status"] == "failed"
    assert terminal_job["completed_at"] is not None

    result = db.operator_retry_job(job_id)
    assert result is not None
    assert result["requeued"] == [task_id]

    retried = db.get_job(job_id)
    assert retried is not None
    assert retried["status"] == "queued"
    assert retried["completed_at"] is None
    assert db.get_queued_tasks()[0]["task_id"] == task_id
