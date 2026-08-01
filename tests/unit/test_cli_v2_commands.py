"""Unit tests for the CLI v2 surface added to scripts/cli/distribai_cli.py:
FleetViewer, CreditsViewer, OrchestratorController, packaging info, the
dashboard opener, and the NodeManager/JobManager methods layered on top of
the shared AdminAPIClient/ManagedProcess/identity helpers.
"""

from __future__ import annotations

from unittest import mock

import pytest

from scripts.cli import distribai_cli
from scripts.cli.distribai_cli import (
    CreditsViewer,
    FleetViewer,
    JobManager,
    NodeManager,
    OrchestratorController,
    main,
    open_dashboard,
    print_packaging_info,
)


@pytest.fixture
def node_manager(tmp_path):
    manager = NodeManager()
    manager.config_dir = tmp_path / ".distribai"
    manager.config_file = manager.config_dir / "desktop.json"
    return manager


# --- FleetViewer -----------------------------------------------------------


@pytest.mark.unit
def test_fleet_viewer_lists_nodes(capsys):
    nodes = [
        {
            "node_id": "node-a",
            "status": "active",
            "online": True,
            "credits": 12.5,
            "hardware_summary": "RTX 3090",
        }
    ]
    with mock.patch.object(distribai_cli.AdminAPIClient, "list_nodes", return_value=nodes):
        FleetViewer().list_nodes()
    out = capsys.readouterr().out
    assert "node-a" in out
    assert "RTX 3090" in out


@pytest.mark.unit
def test_fleet_viewer_handles_empty_fleet(capsys):
    with mock.patch.object(distribai_cli.AdminAPIClient, "list_nodes", return_value=[]):
        FleetViewer().list_nodes()
    assert "No nodes registered" in capsys.readouterr().out


# --- CreditsViewer -----------------------------------------------------------


@pytest.mark.unit
def test_credits_viewer_lists_balances(capsys):
    credits_map = {"node-a": {"balance": 10.0, "lifetime": 25.0}}
    with mock.patch.object(distribai_cli.AdminAPIClient, "list_credits", return_value=credits_map):
        CreditsViewer().list_credits()
    out = capsys.readouterr().out
    assert "node-a" in out
    assert "10.00" in out
    assert "25.00" in out


@pytest.mark.unit
def test_credits_viewer_handles_empty_map(capsys):
    with mock.patch.object(distribai_cli.AdminAPIClient, "list_credits", return_value={}):
        CreditsViewer().list_credits()
    assert "No credit balances found" in capsys.readouterr().out


# --- OrchestratorController --------------------------------------------------


@pytest.mark.unit
def test_orchestrator_controller_start_success(capsys):
    controller = OrchestratorController()
    with mock.patch.object(
        distribai_cli.ManagedProcess, "start", return_value={"ok": True, "pid": 123, "log_file": "x.log"}
    ):
        controller.start(50051, 8766)
    out = capsys.readouterr().out
    assert "started" in out.lower()
    assert "50051" in out
    assert "8766" in out


@pytest.mark.unit
def test_orchestrator_controller_start_failure(capsys):
    controller = OrchestratorController()
    with mock.patch.object(
        distribai_cli.ManagedProcess, "start", return_value={"ok": False, "error": "already running"}
    ):
        controller.start(None, None)
    result = capsys.readouterr()
    assert "already running" in result.out + result.err


@pytest.mark.unit
def test_orchestrator_controller_stop_success(capsys):
    controller = OrchestratorController()
    with mock.patch.object(
        distribai_cli.ManagedProcess, "stop", return_value={"ok": True, "pid": 123}
    ):
        controller.stop()
    assert "stopped" in capsys.readouterr().out.lower()


@pytest.mark.unit
def test_orchestrator_controller_stop_failure(capsys):
    controller = OrchestratorController()
    with mock.patch.object(
        distribai_cli.ManagedProcess, "stop", return_value={"ok": False, "error": "not running"}
    ):
        controller.stop()
    result = capsys.readouterr()
    assert "not running" in result.out + result.err


@pytest.mark.unit
def test_orchestrator_controller_status_reachable(capsys):
    controller = OrchestratorController()
    with (
        mock.patch.object(
            distribai_cli.ManagedProcess, "status", return_value={"running": True, "pid": 42}
        ),
        mock.patch.object(
            distribai_cli.AdminAPIClient,
            "health",
            return_value={"active_nodes": 2, "queued_jobs": 1, "running_jobs": 0},
        ),
    ):
        controller.status(None)
    out = capsys.readouterr().out
    assert "running" in out.lower()
    assert "Active nodes: 2" in out


@pytest.mark.unit
def test_orchestrator_controller_status_unreachable(capsys):
    controller = OrchestratorController()
    with (
        mock.patch.object(
            distribai_cli.ManagedProcess, "status", return_value={"running": False, "pid": None}
        ),
        mock.patch.object(
            distribai_cli.AdminAPIClient, "health", return_value={"error": "offline"}
        ),
    ):
        controller.status(None)
    result = capsys.readouterr()
    combined = (result.out + result.err).lower()
    assert "not running" in combined
    assert "unreachable" in combined


@pytest.mark.unit
def test_orchestrator_controller_logs(capsys):
    controller = OrchestratorController()
    with mock.patch.object(distribai_cli.ManagedProcess, "tail_logs", return_value=["line1", "line2"]):
        controller.logs(10)
    out = capsys.readouterr().out
    assert "line1" in out
    assert "line2" in out


# --- open_dashboard / print_packaging_info -----------------------------------


@pytest.mark.unit
def test_open_dashboard_node_uses_default_url(monkeypatch):
    monkeypatch.delenv("DISTRIBAI_NODE_DASHBOARD_URL", raising=False)
    captured = {}
    monkeypatch.setattr(distribai_cli.webbrowser, "open", lambda url: captured.setdefault("url", url))
    open_dashboard("node")
    assert captured["url"] == "http://127.0.0.1:3000"


@pytest.mark.unit
def test_open_dashboard_orchestrator_respects_env_override(monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ORCH_DASHBOARD_URL", "http://example.test:9999")
    captured = {}
    monkeypatch.setattr(distribai_cli.webbrowser, "open", lambda url: captured.setdefault("url", url))
    open_dashboard("orchestrator")
    assert captured["url"] == "http://example.test:9999"


@pytest.mark.unit
def test_print_packaging_info_covers_every_audience(capsys):
    print_packaging_info()
    out = capsys.readouterr().out
    assert "Community" in out
    assert "Org / operator" in out
    assert "Admin" in out
    assert "specs/node-windows.spec" in out
    assert "specs/server-windows.spec" in out


# --- NodeManager: identity / process lifecycle -------------------------------


@pytest.mark.unit
def test_node_manager_show_identity_generates_and_persists(node_manager):
    identity = node_manager.show_identity()
    assert identity["org_id"].startswith("org-")
    assert identity["node_id"]
    persisted = node_manager.get_config()
    assert persisted["org_id"] == identity["org_id"]


@pytest.mark.unit
def test_node_manager_show_identity_is_stable_across_calls(node_manager):
    first = node_manager.show_identity()
    second = node_manager.show_identity()
    assert first["org_id"] == second["org_id"]
    assert first["node_id"] == second["node_id"]


@pytest.mark.unit
def test_node_manager_set_name_writes_node_name_key(node_manager):
    node_manager.set_name("my-rig")
    config = node_manager.get_config()
    assert config["node_name"] == "my-rig"
    assert "nodeName" not in config


@pytest.mark.unit
def test_node_manager_set_name_migrates_legacy_camelcase_key(node_manager):
    node_manager.set_config({"nodeName": "legacy"})
    node_manager.set_name("renamed")
    config = node_manager.get_config()
    assert config["node_name"] == "renamed"
    assert "nodeName" not in config


@pytest.mark.unit
def test_node_manager_start_delegates_to_managed_process(node_manager, capsys):
    node_manager.set_config({"node_id": "my-node"})
    with mock.patch.object(
        distribai_cli.ManagedProcess,
        "start",
        return_value={"ok": True, "pid": 55, "log_file": "node.log"},
    ) as start_mock:
        node_manager.start("localhost:50051", 1)
    assert start_mock.call_args[0][0] == distribai_cli.worker_argv("localhost:50051", "my-node", 1)
    assert "started" in capsys.readouterr().out.lower()


@pytest.mark.unit
def test_node_manager_stop_and_logs(node_manager, capsys):
    with mock.patch.object(distribai_cli.ManagedProcess, "stop", return_value={"ok": True, "pid": 9}):
        node_manager.stop()
    assert "stopped" in capsys.readouterr().out.lower()

    with mock.patch.object(distribai_cli.ManagedProcess, "tail_logs", return_value=["hello"]):
        node_manager.logs(5)
    assert "hello" in capsys.readouterr().out


@pytest.mark.unit
def test_node_manager_status_shows_identity_and_process_state(node_manager, capsys):
    node_manager.set_config({"node_name": "rig-1", "node_id": "rig-1", "org_id": "org-1"})
    with mock.patch.object(
        distribai_cli.ManagedProcess, "status", return_value={"running": True, "pid": 7}
    ):
        node_manager.status()
    out = capsys.readouterr().out
    assert "rig-1" in out
    assert "org-1" in out
    assert "Running" in out


# --- JobManager: status / watch / create -------------------------------------


@pytest.mark.unit
def test_job_manager_job_status_prints_fields(capsys):
    manager = JobManager()
    job = {"job_id": "j1", "status": "running", "model_name": "m", "steps": 5}
    with mock.patch.object(manager, "_api_call", return_value=job):
        result = manager.job_status("j1")
    assert result == job
    out = capsys.readouterr().out
    assert "running" in out


@pytest.mark.unit
def test_job_manager_job_status_reports_error(capsys):
    manager = JobManager()
    with mock.patch.object(manager, "_api_call", return_value={"error": "not found"}):
        result = manager.job_status("missing")
    assert result == {"error": "not found"}
    captured = capsys.readouterr()
    assert "not found" in captured.out + captured.err


@pytest.mark.unit
def test_job_manager_watch_job_polls_until_terminal(monkeypatch, capsys):
    manager = JobManager()
    statuses = iter(["queued", "running", "success"])
    monkeypatch.setattr(
        manager, "_api_call", lambda *a, **k: {"job_id": "j1", "status": next(statuses)}
    )
    monkeypatch.setattr(distribai_cli.time, "sleep", lambda _seconds: None)
    final = manager.watch_job("j1", poll_seconds=0, timeout_seconds=5)
    assert final == "success"
    out = capsys.readouterr().out
    assert "queued" in out
    assert "running" in out
    assert "success" in out


@pytest.mark.unit
def test_job_manager_watch_job_times_out(monkeypatch, capsys):
    manager = JobManager()
    monkeypatch.setattr(manager, "_api_call", lambda *a, **k: {"job_id": "j1", "status": "running"})
    clock = iter([0.0, 0.1, 10.0])
    monkeypatch.setattr(distribai_cli.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(distribai_cli.time, "sleep", lambda _seconds: None)
    result = manager.watch_job("j1", poll_seconds=0, timeout_seconds=1)
    assert result == "timeout"
    assert "timed out" in capsys.readouterr().out.lower()


@pytest.mark.unit
def test_job_manager_create_job_minimal_body_has_no_optional_keys():
    manager = JobManager()
    with mock.patch.object(manager, "_api_call", return_value={"job_id": "j1"}) as call:
        manager.create_job("distribai-tiny", 10)
    body = call.call_args[0][2]
    assert body == {"model_name": "distribai-tiny", "steps": 10, "batch_size": 32}


@pytest.mark.unit
def test_job_manager_create_job_forwards_every_optional_field():
    manager = JobManager()
    with mock.patch.object(manager, "_api_call", return_value={"job_id": "j1"}) as call:
        manager.create_job(
            "distribai-tiny",
            10,
            batch_size=16,
            org="acme-corp",
            job_type="train",
            priority=5,
            priority_tier="P0",
            submitter_id="alice",
            description="test job",
            deadline_seconds=900,
            steps_per_task=5,
            learning_rate=0.01,
            weight_url="s3://weights",
            batch_url="s3://batches",
        )
    body = call.call_args[0][2]
    assert body == {
        "model_name": "distribai-tiny",
        "steps": 10,
        "batch_size": 16,
        "org": "acme-corp",
        "job_type": "train",
        "priority": 5,
        "priority_tier": "P0",
        "submitter_id": "alice",
        "description": "test job",
        "deadline_seconds": 900,
        "steps_per_task": 5,
        "weight_blob_url": "s3://weights",
        "batch_blob_url": "s3://batches",
        "hparams": {"lr": 0.01},
    }


@pytest.mark.unit
def test_job_manager_create_job_reports_error(capsys):
    manager = JobManager()
    with mock.patch.object(manager, "_api_call", return_value={"error": "quota exceeded"}):
        manager.create_job("distribai-tiny", 10)
    captured = capsys.readouterr()
    assert "quota exceeded" in captured.out + captured.err


# --- main() dispatch wiring ---------------------------------------------------


@pytest.mark.unit
def test_main_dispatches_nodes_list():
    with mock.patch.object(distribai_cli.FleetViewer, "list_nodes") as mocked:
        main(["nodes", "list"])
    mocked.assert_called_once()


@pytest.mark.unit
def test_main_dispatches_credits_list():
    with mock.patch.object(distribai_cli.CreditsViewer, "list_credits") as mocked:
        main(["credits", "list"])
    mocked.assert_called_once()


@pytest.mark.unit
def test_main_dispatches_orchestrator_start():
    with mock.patch.object(distribai_cli.OrchestratorController, "start") as mocked:
        main(["orchestrator", "start", "--grpc-port", "1", "--admin-port", "2"])
    mocked.assert_called_once_with(1, 2)


@pytest.mark.unit
def test_main_dispatches_node_identity():
    with mock.patch.object(distribai_cli.NodeManager, "show_identity") as mocked:
        main(["node", "identity"])
    mocked.assert_called_once()


@pytest.mark.unit
def test_main_dispatches_package_info():
    with mock.patch.object(distribai_cli, "print_packaging_info") as mocked:
        main(["package", "info"])
    mocked.assert_called_once()


@pytest.mark.unit
def test_main_dispatches_dashboard_with_target():
    with mock.patch.object(distribai_cli, "open_dashboard") as mocked:
        main(["dashboard", "orchestrator"])
    mocked.assert_called_once_with("orchestrator")


@pytest.mark.unit
def test_main_dispatches_job_watch():
    with mock.patch.object(distribai_cli.JobManager, "watch_job") as mocked:
        main(["job", "watch", "j1", "--interval", "0.1", "--timeout", "1"])
    mocked.assert_called_once_with("j1", poll_seconds=0.1, timeout_seconds=1.0)


@pytest.mark.unit
def test_main_with_no_command_prints_help_and_exits(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 1
    assert "usage" in capsys.readouterr().out.lower()
