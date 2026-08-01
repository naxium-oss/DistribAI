"""Integration: ScriptRunner unpacks and executes a real tarball (no worker mocks)."""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from worker.src.daemon.script_runner import ScriptRunner


def _minimal_package(run_body: bytes | None = None) -> bytes:
    body = run_body or (
        b"import json\n"
        b'with open("results.json", "w", encoding="utf-8") as f:\n'
        b'    json.dump({"ok": True, "credits_earned": 1}, f)\n'
    )
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="run.py")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    return buf.getvalue()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_runner_executes_minimal_tarball(tmp_path):
    runner = ScriptRunner(work_dir=tmp_path / "jobs")
    result = await runner.execute_task("pkg-task-01", _minimal_package(), {}, {})

    assert result["status"] == "completed"
    assert result.get("results", {}).get("ok") is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_runner_rejects_path_traversal_member(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        evil = b"print('nope')\n"
        info = tarfile.TarInfo(name="../escape.py")
        info.size = len(evil)
        tar.addfile(info, io.BytesIO(evil))
    runner = ScriptRunner(work_dir=tmp_path / "jobs2")
    result = await runner.execute_task("pkg-task-02", buf.getvalue(), {}, {})

    assert result["status"] == "failed"
    assert "Invalid tar member" in result.get("error", "")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_runner_missing_run_py_fails(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"x = 1\n"
        info = tarfile.TarInfo(name="other.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    runner = ScriptRunner(work_dir=tmp_path / "jobs3")
    result = await runner.execute_task("pkg-task-03", buf.getvalue(), {}, {})

    assert result["status"] == "failed"
    assert "run.py" in result.get("error", "")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_runner_reports_sandbox_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_SANDBOX_BACKEND", "subprocess")
    runner = ScriptRunner(work_dir=tmp_path / "jobs6")
    result = await runner.execute_task("pkg-task-06", _minimal_package(), {}, {})

    assert result["status"] == "completed"
    assert result.get("backend_used") == "subprocess"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_runner_preserves_env_job_id_without_config(tmp_path):
    body = (
        b"import json, os\n"
        b'with open("results.json", "w", encoding="utf-8") as f:\n'
        b'    json.dump({"job_id": os.getenv("DISTRIBAI_JOB_ID")}, f)\n'
    )
    runner = ScriptRunner(work_dir=tmp_path / "jobs-env")
    result = await runner.execute_task(
        "pkg-task-env",
        _minimal_package(body),
        {"DISTRIBAI_JOB_ID": "job-env-1"},
        {},
    )

    assert result["status"] == "completed"
    assert result["results"]["job_id"] == "job-env-1"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_runner_rejects_checksum_mismatch(tmp_path):
    pkg = _minimal_package()
    wrong = "0" * 64
    runner = ScriptRunner(work_dir=tmp_path / "jobs4")
    result = await runner.execute_task(
        "pkg-task-04",
        pkg,
        {},
        {"package_sha256": wrong},
    )

    assert result["status"] == "failed"
    assert "checksum mismatch" in result.get("error", "")
    assert hashlib.sha256(pkg).hexdigest() != wrong


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_runner_blocks_pip_when_egress_denied(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_DENY_EGRESS", "true")
    body = b"print('ok')\n"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in (
            ("run.py", body),
            ("requirements.txt", b"numpy\n"),
        ):
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    runner = ScriptRunner(work_dir=tmp_path / "jobs5")
    result = await runner.execute_task("pkg-task-05", buf.getvalue(), {}, {})
    assert result["status"] == "failed"
    assert "egress denied" in result.get("error", "").lower()
