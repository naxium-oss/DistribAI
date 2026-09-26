"""Extended tests for state module."""

import tempfile

import pytest

try:
    from worker.src.daemon.state import AuthTokens, WorkerState

    HAS_STATE = True
except ImportError:
    HAS_STATE = False


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_worker_state_creation():
    """Test WorkerState creation."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        assert state.node_id == "node-1"
        assert state.get_status() == "starting"


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_set_status():
    """Test set_status method."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        state.set_status("idle")
        assert state.get_status() == "idle"


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_set_status_with_job():
    """Test set_status with job info."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        job = {
            "job_id": "job-1",
            "task_id": "task-1",
            "model_name": "test",
            "steps": 100,
            "batch_size": 1,
        }
        state.set_status("working", job)
        assert state.get_status() == "working"


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_update_heartbeat():
    """Test update_heartbeat method."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        state.update_heartbeat(5)
        assert state._state["heartbeat_seq"] == 5


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_record_job_done_success():
    """Test record_job_done with success."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        state.record_job_done("success")
        assert state._state["jobs_completed"] == 1


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_record_job_done_failure():
    """Test record_job_done with failure."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        state.record_job_done("failure")
        assert state._state["jobs_failed"] == 1


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_save_auth_tokens():
    """Test save_auth_tokens method."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        state.save_auth_tokens(jwt_token="test-jwt", session_token="test-session")
        tokens = state.load_auth_tokens()
        assert tokens.jwt_token == "test-jwt"
        assert tokens.session_token == "test-session"


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_load_auth_tokens():
    """Test load_auth_tokens method."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        tokens = state.load_auth_tokens()
        assert tokens is not None


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_clear_auth_tokens():
    """Test clear_auth_tokens method."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        state.save_auth_tokens(jwt_token="test-jwt")
        state.clear_auth_tokens()
        tokens = state.load_auth_tokens()
        assert tokens.jwt_token is None


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_load_benchmark_results():
    """Test load_benchmark_results method."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        result = state.load_benchmark_results()
        # Should return None when no results file exists
        assert result is None


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_auth_tokens_dataclass():
    """Test AuthTokens dataclass."""
    tokens = AuthTokens(jwt_token="jwt", session_token="session", node_id="node-1")
    assert tokens.jwt_token == "jwt"
    assert tokens.session_token == "session"
    assert tokens.node_id == "node-1"


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_state_persistence():
    """Test state persists across reloads."""
    with tempfile.TemporaryDirectory() as tmp:
        state1 = WorkerState(str(tmp), "node-1")
        state1.set_status("idle")
        state1.update_heartbeat(10)

        state2 = WorkerState(str(tmp), "node-1")
        assert state2.get_status() == "idle"
        assert state2._state["heartbeat_seq"] == 10


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_flush_method():
    """Test _flush method."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        state._state["test"] = "value"
        state._flush()

        data = state._load_from_file()
        assert data.get("test") == "value"


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_log_event():
    """Test _log_event method."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        state._log_event("test_event", {"key": "value"})

        assert state.events_file.exists()


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_load_from_corrupted_file():
    """Test _load_from_file handles corrupted file."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        state.state_file.write_text("invalid json")

        data = state._load_from_file()
        assert data == {}


@pytest.mark.skipif(not HAS_STATE, reason="state not available")
def test_save_to_file_atomic():
    """Test _save_to_file writes atomically."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(str(tmp), "node-1")
        test_data = {"test": "value"}
        state._save_to_file(test_data)

        assert state.state_file.exists()
        data = state._load_from_file()
        assert data.get("test") == "value"
