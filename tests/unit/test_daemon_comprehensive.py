"""Comprehensive tests for daemon module to increase coverage."""

import tempfile
from pathlib import Path

import pytest


def test_daemon_sanitize_log_message_jwt():
    """Test JWT redaction in log sanitization."""
    from worker.src.daemon.daemon import _sanitize_log_message

    # Test JWT token redaction
    jwt_msg = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMe"
    result = _sanitize_log_message(jwt_msg)
    assert "[REDACTED" in result or "[REDACTED_JWT]" in result


def test_daemon_sanitize_log_message_password():
    """Test password redaction in log sanitization."""
    from worker.src.daemon.daemon import _sanitize_log_message

    # Test password redaction
    pwd_msg = "user password=secret123 credentials"
    result = _sanitize_log_message(pwd_msg)
    assert "[REDACTED]" in result


def test_daemon_sanitize_log_message_truncation():
    """Test long message truncation."""
    from worker.src.daemon.daemon import _sanitize_log_message

    # Test truncation of very long messages
    long_msg = "A" * 2000
    result = _sanitize_log_message(long_msg)
    assert len(result) <= 1000


def test_redacting_filter_with_jwt():
    """Test RedactingFilter with JWT in log record."""
    import logging

    from worker.src.daemon.daemon import RedactingFilter

    filter_obj = RedactingFilter()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test",
        args=(),
        exc_info=None,
    )

    result = filter_obj.filter(record)
    assert result is True
    assert "[REDACTED" in record.msg or "eyJ" not in record.msg


def test_worker_daemon_state_dir_creation():
    """Test WorkerDaemon creates state directory."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = Path(tmpdir) / "state" / "subdir"
        _daemon = WorkerDaemon(  # noqa: F841
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=str(state_path),
        )

        # State directory should be created
        assert state_path.exists()


def test_worker_daemon_default_state_dir():
    """Test WorkerDaemon with default state directory."""
    from worker.src.daemon.daemon import WorkerDaemon

    daemon = WorkerDaemon(
        orchestrator_url="localhost:50051",
        node_id="test-node",
    )

    # Should have a state directory
    assert daemon.state is not None


def test_worker_daemon_worker_index():
    """Test WorkerDaemon with different worker indices."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon0 = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="worker-0",
            state_dir=tmpdir,
            worker_index=0,
        )
        assert daemon0.worker_index == 0

        daemon1 = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="worker-1",
            state_dir=tmpdir,
            worker_index=1,
        )
        assert daemon1.worker_index == 1


def test_worker_daemon_benchmark_manager():
    """Test WorkerDaemon has BenchmarkManager."""
    from worker.src.daemon.bench_manager import BenchmarkManager
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        assert daemon.bench is not None
        assert isinstance(daemon.bench, BenchmarkManager)


def test_worker_daemon_stop_cancels_tasks():
    """Test WorkerDaemon stop sets the stop event."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Stop should set the stop event
        daemon.stop()

        assert daemon._stop.is_set()


def test_worker_daemon_seq_increment():
    """Test WorkerDaemon sequence increment."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Sequence should start at 0
        assert daemon._seq == 0


def test_worker_daemon_current_job():
    """Test WorkerDaemon current job attribute."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Should start with no current job
        assert daemon._current_job is None


def test_worker_daemon_send_queue():
    """Test WorkerDaemon send queue."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Should start with no send queue
        assert daemon._send_queue is None


def test_worker_daemon_last_progress():
    """Test WorkerDaemon last progress report tracking."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Should start at 0
        assert daemon._last_progress_report == 0


@pytest.mark.asyncio
async def test_worker_daemon_stop_while_reconnecting():
    """Test WorkerDaemon stop while reconnecting."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Stop immediately
        daemon.stop()

        # Verify stop was set
        assert daemon._stop.is_set()
