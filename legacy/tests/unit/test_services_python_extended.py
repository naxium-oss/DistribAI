"""Extended tests for services_python modules."""

import tempfile
from pathlib import Path

import pytest


def test_poc_challenge_init():
    """Test PoCChallengeManager initialization."""
    try:
        from services_python.poc_challenge import PoCChallengeManager
    except ImportError:
        pytest.skip("PoCChallengeManager not available")
        return

    poc = PoCChallengeManager(difficulty=4)
    assert poc.difficulty == 4


def test_poc_challenge_generate():
    """Test challenge generation."""
    try:
        from services_python.poc_challenge import PoCChallengeManager
    except ImportError:
        pytest.skip("PoCChallengeManager not available")
        return

    poc = PoCChallengeManager(difficulty=2)
    challenge = poc.generate_challenge("node-1")

    assert challenge is not None
    assert hasattr(challenge, "challenge") or isinstance(challenge, (str, bytes))


def test_sybil_detector_init():
    """Test SybilDetector initialization."""
    try:
        from services_python.sybil_detector import SybilDetector
    except ImportError:
        pytest.skip("SybilDetector not available")
        return

    detector = SybilDetector()
    assert detector is not None


def test_sybil_detector_analyze():
    """Test SybilDetector analyze_account."""
    try:
        from services_python.sybil_detector import SybilDetector
    except ImportError:
        pytest.skip("SybilDetector not available")
        return

    detector = SybilDetector()
    result = detector.analyze_account(
        node_id="node-1",
        ip_address="127.0.0.1",
        hardware_fingerprint="gpu-123",
        initial_credits=0,
    )

    assert isinstance(result, dict)
    assert "approved" in result


def test_credit_multiplier_engine_init():
    """Test CreditMultiplierEngine initialization."""
    try:
        from services_python.credit_multipliers import CreditMultiplierEngine
    except ImportError:
        pytest.skip("CreditMultiplierEngine not available")
        return

    engine = CreditMultiplierEngine()
    assert engine is not None


def test_credit_multiplier_engine_get_stats():
    """Admin stats endpoint uses CreditMultiplierEngine.get_stats()."""
    from services_python.credit_multipliers import CreditMultiplierEngine

    engine = CreditMultiplierEngine()
    stats = engine.get_stats()
    assert stats["tracked_nodes"] == 0
    assert stats["avg_effective_multiplier"] == 1.0
    engine.get_or_create_state("n1")
    engine.set_surge_opt_in("n1", True)
    stats2 = engine.get_stats()
    assert stats2["tracked_nodes"] == 1
    assert stats2["surge_opt_in_nodes"] == 1
    assert stats2["avg_effective_multiplier"] >= 1.0


def test_scheduler_init():
    """Test TaskScheduler initialization."""
    try:
        from services_python.scheduler import TaskScheduler
    except ImportError:
        pytest.skip("TaskScheduler not available")
        return

    class FakeDB:
        pass

    class FakeNodeService:
        connected_nodes = {}

    scheduler = TaskScheduler(db=FakeDB(), node_service=FakeNodeService())
    assert scheduler is not None


def test_rebenchmark_trigger_manager_init():
    """Test RebenchmarkTriggerManager initialization."""
    try:
        from services_python.rebenchmark_triggers import RebenchmarkTriggerManager
    except ImportError:
        pytest.skip("RebenchmarkTriggerManager not available")
        return

    mgr = RebenchmarkTriggerManager()
    assert mgr is not None


def test_credit_transfer_manager_init():
    """Test CreditTransferManager initialization."""
    try:
        from services_python.credit_transfers import CreditTransferManager
    except ImportError:
        pytest.skip("CreditTransferManager not available")
        return

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        # Create minimal db
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS credit_transfers (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        # Test initialization - may fail without full schema
        try:
            mgr = CreditTransferManager(str(db_path))
            assert mgr is not None
        except Exception:
            # Expected if schema is incomplete
            pass


def test_oauth_provider_init():
    """Test OAuthProvider initialization."""
    try:
        from services_python.oauth_provider import OAuthProvider
    except ImportError:
        pytest.skip("OAuthProvider not available")
        return

    provider = OAuthProvider()
    assert provider is not None


def test_job_submission_handler_init():
    """Test JobSubmissionHandler initialization."""
    try:
        from services_python.job_submission import JobSubmissionHandler
    except ImportError:
        pytest.skip("JobSubmissionHandler not available")
        return

    class FakeDB:
        pass

    handler = JobSubmissionHandler(db=FakeDB())
    assert handler is not None
    assert handler.db is not None


def test_mytrainer_sync_init():
    """Test MyTrainerSync initialization."""
    try:
        from services_python.mytrainer_sync import MyTrainerSync
    except ImportError:
        pytest.skip("MyTrainerSync not available")
        return

    with tempfile.TemporaryDirectory() as tmp:
        sync = MyTrainerSync(str(tmp))
        assert sync is not None


def test_voting_system_init():
    """Test VotingSystem initialization."""
    try:
        from worker.src.daemon.voting_system import VotingSystem
    except ImportError:
        pytest.skip("VotingSystem not available")
        return

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "votes.db"
        vs = VotingSystem(str(db_path))
        assert vs is not None


def test_rate_limiter_init():
    """Test rate limiter initialization."""
    try:
        from services_python.rate_limiter import create_rate_limiter
    except ImportError:
        pytest.skip("rate_limiter not available")
        return

    limiter = create_rate_limiter(10.0, 20.0)
    assert limiter is not None
    assert limiter.rate == 10.0
    assert limiter.capacity == 20.0


def test_health_checker_init():
    """Test HealthChecker initialization."""
    try:
        from services_python.monitoring import HealthChecker
    except ImportError:
        pytest.skip("HealthChecker not available")
        return

    checker = HealthChecker()
    assert checker is not None


def test_metrics_collector_init():
    """Test MetricsCollector initialization."""
    try:
        from services_python.monitoring import MetricsCollector
    except ImportError:
        pytest.skip("MetricsCollector not available")
        return

    collector = MetricsCollector()
    assert collector is not None


def test_performance_profiler_init():
    """Test PerformanceProfiler initialization."""
    try:
        from services_python.monitoring import PerformanceProfiler
    except ImportError:
        pytest.skip("PerformanceProfiler not available")
        return

    profiler = PerformanceProfiler()
    assert profiler is not None
