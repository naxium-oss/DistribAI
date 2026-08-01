"""Tests for worker daemon module."""

import tempfile

import pytest

try:
    from worker.src.daemon.state import WorkerState

    HAS_DAEMON = True
except ImportError:
    HAS_DAEMON = False


try:
    from worker.src.daemon.registration import RegistrationManager

    HAS_REGISTRATION = True
except ImportError:
    HAS_REGISTRATION = False
    RegistrationManager = None


try:
    from worker.src.daemon.s3_util import S3Manager

    HAS_S3 = True
except ImportError:
    HAS_S3 = False


@pytest.mark.skipif(not HAS_DAEMON, reason="daemon not available")
def test_worker_state_creation():
    """Test WorkerState can be created."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(tmp, "test-node")
        assert state is not None
        assert state.node_id == "test-node"


@pytest.mark.skipif(not HAS_DAEMON, reason="daemon not available")
def test_worker_state_get_status():
    """Test WorkerState get_status."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(tmp, "test-node")
        status = state.get_status()
        assert status == "starting"


@pytest.mark.skipif(not HAS_DAEMON, reason="daemon not available")
def test_worker_state_auth_tokens():
    """Test WorkerState auth token management."""
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(tmp, "test-node")
        state.save_auth_tokens(jwt_token="jwt-123", session_token="session-456")
        tokens = state.load_auth_tokens()
        assert tokens.jwt_token == "jwt-123"
        assert tokens.session_token == "session-456"
        assert tokens.node_id == "test-node"


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_registration_manager_creation():
    """Test RegistrationManager can be created."""
    registrar = RegistrationManager(
        orchestrator_admin_url="http://localhost:8766",
        node_id="test-node",
    )
    assert registrar is not None
    assert registrar.node_id == "test-node"


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_registration_manager_detect_hardware():
    """Test RegistrationManager hardware detection."""
    registrar = RegistrationManager(
        orchestrator_admin_url="http://localhost:8766",
        node_id="test-node",
    )
    hardware = registrar.detect_hardware()
    assert "node_id" in hardware
    assert "os" in hardware
    assert "cpu_count" in hardware
    assert "ram_total_gb" in hardware


@pytest.mark.skipif(not HAS_S3, reason="s3_util not available")
def test_s3_manager_creation():
    """Test S3Manager creation."""
    manager = S3Manager()
    assert manager is not None


@pytest.mark.skipif(not HAS_S3, reason="s3_util not available")
def test_s3_manager_parse_s3_url():
    """Test S3Manager URL parsing."""
    manager = S3Manager()
    bucket, key = manager._parse_s3_url("s3://mybucket/path/to/file.pt")
    assert bucket == "mybucket"
    assert key == "path/to/file.pt"


@pytest.mark.skipif(not HAS_S3, reason="s3_util not available")
def test_s3_manager_parse_s3_url_invalid():
    """Test S3Manager URL parsing with invalid URL."""
    manager = S3Manager()
    with pytest.raises(ValueError):
        manager._parse_s3_url("http://example.com/file.pt")


@pytest.mark.skipif(not HAS_S3, reason="s3_util not available")
def test_s3_manager_is_s3_url():
    """Test S3Manager URL type detection."""
    manager = S3Manager()
    assert manager._is_s3_url("s3://bucket/key") is True
    assert manager._is_s3_url("http://example.com") is False
    assert manager._is_s3_url("/local/path") is False
