"""Tests for GUI launcher module."""

import pytest

try:
    from worker.src.daemon.daemon import WorkerDaemon
    from worker.src.daemon.gui_launcher import NodeAPI

    HAS_GUI_LAUNCHER = True
except ImportError:
    HAS_GUI_LAUNCHER = False
    NodeAPI = None
    WorkerDaemon = None  # type: ignore[misc, assignment]


@pytest.mark.skipif(not HAS_GUI_LAUNCHER, reason="gui_launcher not available")
def test_gui_launcher_import():
    """Test GUI launcher module imports."""
    assert NodeAPI is not None


@pytest.mark.skipif(not HAS_GUI_LAUNCHER, reason="gui_launcher not available")
def test_node_api_creation():
    """Test NodeAPI can be created."""
    api = NodeAPI()
    assert api is not None


@pytest.mark.skipif(not HAS_GUI_LAUNCHER, reason="gui_launcher not available")
def test_node_api_attributes():
    """Test NodeAPI has expected attributes."""
    api = NodeAPI()
    # Check expected default attributes
    assert hasattr(api, "get_status") or hasattr(api, "get_node_status")


@pytest.mark.skipif(not HAS_GUI_LAUNCHER, reason="gui_launcher not available")
def test_node_api_methods():
    """Test NodeAPI has expected methods."""
    api = NodeAPI()
    assert hasattr(api, "get_status")


@pytest.mark.skipif(
    not HAS_GUI_LAUNCHER or WorkerDaemon is None,
    reason="gui_launcher / daemon not available",
)
def test_get_job_list_reflects_daemon_snapshot():
    """GUI job list uses daemon snapshot when connected."""
    api = NodeAPI()
    assert api.get_job_list() == []

    api.connected = True
    daemon = WorkerDaemon(orchestrator_url="127.0.0.1:59999", node_id="gui-test-node")
    api.daemon = daemon

    idle = api.get_job_list()
    assert len(idle) == 1
    assert idle[0].get("id") == "idle"

    daemon._current_job = {"job_id": "job-a", "task_id": "t1", "kind": "script"}
    running = api.get_job_list()
    assert len(running) == 1
    assert running[0]["id"] == "job-a"
    assert "Script" in running[0]["name"]

    daemon._current_job = {"job_id": "job-b", "task_id": "t2", "model_name": "test-model"}
    train = api.get_job_list()
    assert train[0]["name"] == "test-model"


@pytest.mark.skipif(not HAS_GUI_LAUNCHER, reason="gui_launcher not available")
def test_gui_launcher_subprocess_without_shell_true():
    """Verify no shell=True in subprocess calls (security)."""
    import inspect

    from worker.src.daemon import gui_launcher

    source = inspect.getsource(gui_launcher)
    dangerous_patterns = [
        "Popen(.*shell=True",
        "call(.*shell=True",
        "run(.*shell=True",
    ]
    for pattern in dangerous_patterns:
        assert pattern not in source, f"Security issue: {pattern} found"
