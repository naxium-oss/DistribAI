"""Extended tests for worker daemon to increase coverage."""

import asyncio
import os
import tempfile

import pytest


def test_sanitize_log_message():
    """Test _sanitize_log_message function."""
    from worker.src.daemon.daemon import _sanitize_log_message

    # Test basic sanitization
    result = _sanitize_log_message("Normal message")
    assert result == "Normal message"

    # Test JWT redaction
    msg_with_jwt = "Token: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.test"
    result = _sanitize_log_message(msg_with_jwt)
    assert "[REDACTED" in result

    # Test empty message
    assert _sanitize_log_message("") == ""


def test_redacting_filter():
    """Test RedactingFilter class."""
    import logging

    from worker.src.daemon.daemon import RedactingFilter

    filter_obj = RedactingFilter()

    # Create a log record
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message with password=secret123",
        args=(),
        exc_info=None,
    )

    # Filter should return True and sanitize
    result = filter_obj.filter(record)
    assert result is True
    assert "[REDACTED]" in record.msg


def test_get_jwt_token():
    """Test _get_jwt_token function."""
    from worker.src.daemon.daemon import _get_jwt_token

    # Test with no env var
    original = os.environ.pop("DISTRIBAI_JWT_TOKEN", None)
    result = _get_jwt_token()
    assert result is None

    # Test with env var set
    os.environ["DISTRIBAI_JWT_TOKEN"] = "test-jwt-token"
    result = _get_jwt_token()
    assert result == "test-jwt-token"

    # Cleanup
    if original:
        os.environ["DISTRIBAI_JWT_TOKEN"] = original
    else:
        os.environ.pop("DISTRIBAI_JWT_TOKEN", None)


def test_worker_daemon_constants():
    """Test WorkerDaemon constants."""
    from worker.src.daemon.daemon import WorkerDaemon

    assert WorkerDaemon.HEARTBEAT_INTERVAL == 10
    assert WorkerDaemon.RECONNECT_DELAY == 5
    assert WorkerDaemon.MAX_RECONNECT_DELAY == 60


def test_worker_daemon_init():
    """Test WorkerDaemon initialization."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        _daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node-1",
            state_dir=tmpdir,
        )
        assert _daemon.orchestrator_url == "localhost:50051"
        assert _daemon.node_id == "test-node-1"
        assert _daemon.worker_index == 0
        assert _daemon._session_token is None
        assert _daemon.connected is False


def test_worker_daemon_node_id_generation():
    """Test WorkerDaemon generates node_id if not provided."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            state_dir=tmpdir,
        )
        assert daemon.node_id is not None
        assert len(daemon.node_id) > 0
        # Node ID could be any format, just check it exists


def test_worker_daemon_url_cleanup():
    """Test WorkerDaemon cleans up URL."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test http:// removal
        daemon1 = WorkerDaemon(
            orchestrator_url="http://localhost:50051",
            state_dir=tmpdir,
        )
        assert daemon1.orchestrator_url == "localhost:50051"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test ws:// removal
        daemon2 = WorkerDaemon(
            orchestrator_url="ws://localhost:50051",
            state_dir=tmpdir,
        )
        assert daemon2.orchestrator_url == "localhost:50051"


def test_worker_daemon_stop():
    """Test WorkerDaemon stop method."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Stop should set the event
        daemon.stop()
        assert daemon._stop.is_set()


@pytest.mark.asyncio
async def test_worker_daemon_dispatches_benchmark_control(monkeypatch):
    from worker.src.daemon.daemon import WorkerDaemon
    from worker.src.distribai_proto import distribai_pb2

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )
        dispatched = False

        async def fake_benchmark():
            nonlocal dispatched
            dispatched = True

        monkeypatch.setattr(daemon, "_run_requested_benchmark", fake_benchmark)
        await daemon._handle_control(distribai_pb2.ControlMessage(action="benchmark"))
        await daemon._benchmark_task
        assert dispatched is True


@pytest.mark.asyncio
async def test_worker_daemon_queues_benchmark_during_startup(monkeypatch):
    from worker.src.daemon.daemon import WorkerDaemon
    from worker.src.distribai_proto import distribai_pb2

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )
        daemon.connected = True
        startup_started = asyncio.Event()
        release_startup = asyncio.Event()
        requested_runs = 0

        async def fake_startup_suite():
            startup_started.set()
            await release_startup.wait()
            return {"overall_score": 1.0}

        async def fake_submit(*_args, **_kwargs):
            return None

        async def fake_requested_benchmark():
            nonlocal requested_runs
            requested_runs += 1

        monkeypatch.setattr(daemon.bench, "run_full_suite", fake_startup_suite)
        monkeypatch.setattr(daemon, "_submit_benchmark_report", fake_submit)
        monkeypatch.setattr(daemon, "_run_requested_benchmark", fake_requested_benchmark)

        startup_task = asyncio.create_task(daemon._run_startup_benchmark({}))
        daemon._benchmark_task = startup_task
        await startup_started.wait()
        await daemon._handle_control(distribai_pb2.ControlMessage(action="benchmark"))
        assert daemon._benchmark_requested is True

        release_startup.set()
        await startup_task
        await asyncio.sleep(0)
        assert requested_runs == 1
        assert daemon._benchmark_requested is False


@pytest.mark.asyncio
async def test_worker_daemon_defers_benchmark_while_busy(monkeypatch):
    from worker.src.daemon.daemon import WorkerDaemon
    from worker.src.distribai_proto import distribai_pb2

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )
        daemon._current_job = {"job_id": "job-1", "kind": "train"}
        called = False

        async def fake_benchmark():
            nonlocal called
            called = True

        monkeypatch.setattr(daemon, "_run_requested_benchmark", fake_benchmark)
        await daemon._handle_control(distribai_pb2.ControlMessage(action="benchmark"))

        assert daemon._benchmark_task is None
        assert daemon._benchmark_requested is True
        assert called is False


@pytest.mark.asyncio
async def test_worker_daemon_run_early_exit():
    """Test WorkerDaemon run exits early when stopped."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Stop immediately so run() exits
        daemon.stop()

        # Run should complete quickly since _stop is set
        await daemon.run()



def test_worker_daemon_state_integration():
    """Test WorkerDaemon integrates with WorkerState."""
    from worker.src.daemon.daemon import WorkerDaemon
    from worker.src.daemon.state import WorkerState

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # State should be initialized
        assert daemon.state is not None
        assert isinstance(daemon.state, WorkerState)
        assert daemon.state.node_id == "test-node"


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

        # Benchmark manager should be initialized
        assert daemon.bench is not None
        assert isinstance(daemon.bench, BenchmarkManager)
        assert daemon.bench.node_id == "test-node"


def test_worker_daemon_executor_lazy_load():
    """Test WorkerDaemon lazy loads executor."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Initially None
        assert daemon.executor is None

        # Get executor should create it
        executor = daemon._get_executor()
        assert executor is not None
        assert daemon.executor is executor


def test_worker_daemon_seq_increment():
    """Test WorkerDaemon sequence number."""
    from worker.src.daemon.daemon import WorkerDaemon

    with tempfile.TemporaryDirectory() as tmpdir:
        daemon = WorkerDaemon(
            orchestrator_url="localhost:50051",
            node_id="test-node",
            state_dir=tmpdir,
        )

        # Initial sequence should be 0
        assert daemon._seq == 0
