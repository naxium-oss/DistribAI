"""Unit tests for scripts.cli.process_manager.ManagedProcess.

Uses a tmp_path-scoped RUN_DIR (monkeypatched) so these tests never touch a
developer's real ~/.distribai/run/ directory, and spawns one short-lived real
subprocess to exercise the actual start/status/stop/log lifecycle end to end
(this is the module under test's own collaborator, not a mock standing in
for the orchestrator or WorkerDaemon).
"""

from __future__ import annotations

import json
import time

import pytest

from scripts.cli import process_manager
from scripts.cli.process_manager import (
    ManagedProcess,
    orchestrator_argv,
    orchestrator_env,
    worker_argv,
)


@pytest.fixture(autouse=True)
def _isolated_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(process_manager, "RUN_DIR", tmp_path / "run")
    yield


@pytest.mark.unit
def test_status_when_never_started():
    info = ManagedProcess("nonexistent-proc").status()
    assert info == {"running": False, "pid": None}


@pytest.mark.unit
def test_status_with_corrupt_pidfile(tmp_path):
    proc = ManagedProcess("corrupt")
    process_manager._pid_file("corrupt").write_text("not json", encoding="utf-8")
    info = proc.status()
    assert info == {"running": False, "pid": None}


@pytest.mark.unit
def test_status_removes_stale_pidfile_for_dead_pid(monkeypatch):
    proc = ManagedProcess("stale")
    process_manager._pid_file("stale").write_text(
        json.dumps({"pid": 999999, "started_at": 1, "command": ["x"]}), encoding="utf-8"
    )
    monkeypatch.setattr(process_manager, "_pid_alive", lambda pid: False)
    info = proc.status()
    assert info["running"] is False
    assert not process_manager._pid_file("stale").exists()


@pytest.mark.unit
def test_stop_when_not_running_returns_error():
    result = ManagedProcess("never-started").stop()
    assert result == {"ok": False, "error": "never-started is not running"}


@pytest.mark.unit
def test_tail_logs_returns_empty_list_when_no_log_file():
    assert ManagedProcess("no-logs").tail_logs() == []


@pytest.mark.unit
def test_start_refuses_when_already_running(monkeypatch):
    proc = ManagedProcess("dup")
    monkeypatch.setattr(proc, "status", lambda: {"running": True, "pid": 123})
    result = proc.start(["-c", "pass"])
    assert result["ok"] is False
    assert "already running" in result["error"]


@pytest.mark.unit
def test_real_process_start_status_logs_stop(tmp_path):
    """End-to-end lifecycle against a real short-lived Python subprocess."""
    proc = ManagedProcess("smoke")
    script = (
        "import sys, time\n"
        "print('hello from child', flush=True)\n"
        "sys.stderr.write('warn line\\n')\n"
        "time.sleep(20)\n"
    )
    start_result = proc.start(["-c", script])
    assert start_result["ok"] is True
    assert isinstance(start_result["pid"], int)

    try:
        deadline = time.monotonic() + 10
        status = proc.status()
        while status["running"] and time.monotonic() < deadline:
            log_lines = proc.tail_logs()
            if any("hello from child" in line for line in log_lines):
                break
            time.sleep(0.5)
            status = proc.status()

        status = proc.status()
        assert status["running"] is True
        assert status["pid"] == start_result["pid"]
        assert status["command"] == ["-c", script]

        log_lines = proc.tail_logs()
        assert any("hello from child" in line for line in log_lines)
    finally:
        stop_result = proc.stop(timeout=5.0)
        assert stop_result["ok"] is True

    assert proc.status()["running"] is False


@pytest.mark.unit
def test_start_passes_env_overrides(tmp_path):
    proc = ManagedProcess("env-check")
    marker = tmp_path / "env_seen.txt"
    script = (
        "import os\n"
        f"open(r'{marker}', 'w').write(os.environ.get('DISTRIBAI_TEST_MARKER', 'missing'))\n"
    )
    result = proc.start(["-c", script], env={"DISTRIBAI_TEST_MARKER": "present"})
    assert result["ok"] is True
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "present"
    # Process exits on its own; give status() a chance to reap the stale pidfile.
    deadline = time.monotonic() + 5
    while proc.status()["running"] and time.monotonic() < deadline:
        time.sleep(0.2)


@pytest.mark.unit
def test_orchestrator_argv_and_env():
    assert orchestrator_argv(None, None) == ["-m", "services_python.orchestrator_grpc"]
    assert orchestrator_env(None, None) == {}
    assert orchestrator_env(50999, 8999) == {"GRPC_PORT": "50999", "ADMIN_PORT": "8999"}


@pytest.mark.unit
def test_worker_argv_variants():
    assert worker_argv(None, None, None) == ["-m", "worker.src.daemon.run"]
    argv = worker_argv("localhost:50051", "my-node", 2)
    assert argv == [
        "-m",
        "worker.src.daemon.run",
        "--orchestrator",
        "localhost:50051",
        "--node-id",
        "my-node",
        "--worker-index",
        "2",
    ]


@pytest.mark.unit
def test_pid_alive_false_for_impossible_pid():
    # -1 is never a real PID on POSIX or Windows (unlike 0, which Windows
    # reports as the always-present System Idle Process).
    assert process_manager._pid_alive(-1) is False
