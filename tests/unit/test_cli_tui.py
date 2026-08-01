"""Smoke + behavior tests for the Textual TUI (scripts/cli/tui.py).

Mounts the real ``DistribAITUI`` app under Textual's headless test harness
(``App.run_test()``), stubbing only the network boundary (``AdminAPIClient``)
and the background-process boundary (``ManagedProcess``) so tests never
depend on a live orchestrator or mutate real ``~/.distribai`` state.
"""

from __future__ import annotations

from unittest import mock

import pytest
from textual.widgets import DataTable, Input, Static

from scripts.cli import tui
from scripts.cli.api_client import AdminAPIClient
from scripts.cli.tui import DistribAITUI, NewJobModal


def _mock_client(**overrides) -> mock.Mock:
    client = mock.Mock(spec=AdminAPIClient)
    client.base_url = "http://127.0.0.1:8766"
    client.health.return_value = overrides.get("health", {"error": "connection refused"})
    client.list_nodes.return_value = overrides.get("nodes", [])
    client.list_jobs.return_value = overrides.get("jobs", [])
    client.list_credits.return_value = overrides.get("credits", {})
    client.tail_logs.return_value = overrides.get("logs", [])
    client.post.return_value = overrides.get("post", {})
    client.delete.return_value = overrides.get("delete", {})
    return client


def _make_app(tmp_path, **client_overrides) -> DistribAITUI:
    app = DistribAITUI()
    app.client = _mock_client(**client_overrides)
    app.node_manager.config_dir = tmp_path / ".distribai"
    app.node_manager.config_file = app.node_manager.config_dir / "desktop.json"
    return app


@pytest.fixture(autouse=True)
def _stub_managed_process():
    """Never touch a real PID/log file under the developer's ~/.distribai/run/."""
    with mock.patch.object(tui.ManagedProcess, "status", return_value={"running": False, "pid": None}):
        yield


@pytest.mark.unit
async def test_tui_mounts_with_all_tabs(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        tab_ids = [tab.id for tab in app.query(tui.TabPane)]
        assert tab_ids == ["overview", "nodes", "jobs", "credits", "settings", "logs"]


@pytest.mark.unit
async def test_tui_overview_shows_unreachable_when_admin_api_down(tmp_path):
    app = _make_app(tmp_path, health={"error": "connection refused"})
    async with app.run_test() as pilot:
        await pilot.pause()
        health_text = app.screen.query_one("#metric-health", Static).content
        assert "unreachable" in str(health_text).lower()


@pytest.mark.unit
async def test_tui_overview_shows_reachable_metrics(tmp_path):
    app = _make_app(
        tmp_path, health={"active_nodes": 3, "queued_jobs": 1, "running_jobs": 2}
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        health_text = str(app.screen.query_one("#metric-health", Static).content).lower()
        nodes_text = str(app.screen.query_one("#metric-nodes", Static).content)
        jobs_text = str(app.screen.query_one("#metric-jobs", Static).content)
        assert "reachable" in health_text
        assert "3" in nodes_text
        assert "1" in jobs_text and "2" in jobs_text


@pytest.mark.unit
async def test_tui_nodes_table_populated(tmp_path):
    nodes = [
        {
            "node_id": "node-a",
            "status": "active",
            "online": True,
            "credits": 12.5,
            "hardware_summary": "generic-gpu-node",
        }
    ]
    app = _make_app(tmp_path, nodes=nodes)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.screen.query_one("#nodes-table", DataTable)
        assert table.row_count == 1


@pytest.mark.unit
async def test_tui_jobs_table_populated(tmp_path):
    jobs = [{"job_id": "job-1", "status": "running", "model_name": "arch-family-a", "steps": 100}]
    app = _make_app(tmp_path, jobs=jobs)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.screen.query_one("#jobs-table", DataTable)
        assert table.row_count == 1
        assert app._job_ids_by_row == ["job-1"]


@pytest.mark.unit
async def test_tui_credits_table_populated(tmp_path):
    credits = {"node-a": {"balance": 10.0, "lifetime": 25.0}}
    app = _make_app(tmp_path, credits=credits)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.screen.query_one("#credits-table", DataTable)
        assert table.row_count == 1


@pytest.mark.unit
async def test_tui_settings_shows_identity(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        detail = str(app.screen.query_one("#settings-detail", Static).content)
        assert "Org ID" in detail
        assert "Node ID" in detail


@pytest.mark.unit
async def test_tui_logs_tab_calls_tail_logs(tmp_path):
    app = _make_app(tmp_path, logs=["line one", "line two"])
    async with app.run_test() as pilot:
        await pilot.pause()
        app.client.tail_logs.assert_called_with(200)


@pytest.mark.unit
async def test_tui_new_job_modal_cancel_dismisses_without_posting(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, NewJobModal)
        await pilot.click("#cancel")
        await pilot.pause()
        app.client.post.assert_not_called()


@pytest.mark.unit
async def test_tui_new_job_modal_submit_posts_job(tmp_path):
    app = _make_app(tmp_path, post={"job_id": "job-new"})
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, NewJobModal)
        app.screen.query_one("#model", Input).value = "arch-family-x"
        await pilot.click("#submit")
        await pilot.pause()
        app.client.post.assert_called_once_with(
            "/admin/jobs", {"model_name": "arch-family-x", "steps": 100, "batch_size": 32}
        )


@pytest.mark.unit
async def test_tui_new_job_modal_reports_error(tmp_path):
    app = _make_app(tmp_path, post={"error": "quota exceeded"})
    notifications = []
    async with app.run_test() as pilot:
        app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
        await pilot.pause()
        await pilot.press("n")
        await pilot.pause()
        await pilot.click("#submit")
        await pilot.pause()
        assert any("quota exceeded" in message for message, _ in notifications)


@pytest.mark.unit
async def test_tui_cancel_job_without_jobs_warns(tmp_path):
    app = _make_app(tmp_path, jobs=[])
    notifications = []
    async with app.run_test() as pilot:
        app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        app.client.delete.assert_not_called()
        assert any(kwargs.get("severity") == "warning" for _, kwargs in notifications)


@pytest.mark.unit
async def test_tui_cancel_job_with_selection_deletes(tmp_path):
    jobs = [{"job_id": "job-1", "status": "running", "model_name": "arch-family-a", "steps": 100}]
    app = _make_app(tmp_path, jobs=jobs)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        app.client.delete.assert_called_once_with("/admin/jobs/job-1")


@pytest.mark.unit
async def test_tui_toggle_orchestrator_starts_when_not_running(tmp_path):
    app = _make_app(tmp_path)
    with mock.patch.object(
        tui.ManagedProcess, "start", return_value={"ok": True, "pid": 4242}
    ) as mock_start:
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
    mock_start.assert_called_once()


@pytest.mark.unit
async def test_tui_toggle_orchestrator_stops_when_running(tmp_path):
    app = _make_app(tmp_path)
    with (
        mock.patch.object(tui.ManagedProcess, "status", return_value={"running": True, "pid": 99}),
        mock.patch.object(tui.ManagedProcess, "stop", return_value={"ok": True, "pid": 99}) as mock_stop,
    ):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("s")
            await pilot.pause()
    mock_stop.assert_called_once()


@pytest.mark.unit
async def test_tui_refresh_binding_triggers_refresh_all(tmp_path):
    app = _make_app(tmp_path)
    async with app.run_test() as pilot:
        await pilot.pause()
        calls_before = app.client.health.call_count
        await pilot.press("r")
        await pilot.pause()
    assert app.client.health.call_count > calls_before


@pytest.mark.unit
def test_run_tui_constructs_app_and_calls_run():
    with mock.patch.object(tui.DistribAITUI, "run") as mock_run:
        tui.run_tui("http://example.test:1234")
    mock_run.assert_called_once_with()


@pytest.mark.unit
def test_main_parses_admin_url_and_delegates_to_run_tui():
    with mock.patch.object(tui, "run_tui") as mock_run_tui:
        tui.main(["--admin-url", "http://example.test:9999"])
    mock_run_tui.assert_called_once_with("http://example.test:9999")


@pytest.mark.unit
def test_main_defaults_admin_url_to_none():
    with mock.patch.object(tui, "run_tui") as mock_run_tui:
        tui.main([])
    mock_run_tui.assert_called_once_with(None)
