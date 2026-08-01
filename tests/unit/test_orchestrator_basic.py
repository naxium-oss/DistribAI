"""Basic tests for orchestrator modules (no heavy dependencies)."""

import pytest


def test_constants_import():
    """Test constants module imports."""
    from services_python.constants import (
        DEFAULT_ADMIN_HOST,
        DEFAULT_ADMIN_PORT,
        DEFAULT_GRPC_PORT,
        JWT_ALGORITHM,
    )

    assert DEFAULT_ADMIN_HOST is not None
    assert isinstance(DEFAULT_ADMIN_PORT, str)
    assert isinstance(DEFAULT_GRPC_PORT, str)
    assert len(DEFAULT_ADMIN_PORT) > 0
    assert len(DEFAULT_GRPC_PORT) > 0
    assert JWT_ALGORITHM is not None


def test_scheduler_import():
    """Test scheduler module imports."""
    try:
        from services_python.scheduler import TaskScheduler

        assert TaskScheduler is not None
    except ImportError:
        pytest.skip("scheduler dependencies not available")


def test_db_manager_import():
    """Test DB manager imports."""
    try:
        from services_python.db_manager import DBManager

        assert DBManager is not None
    except ImportError:
        pytest.skip("db_manager dependencies not available")


def test_job_submission_import():
    """Test job submission imports."""
    try:
        from services_python.job_submission import JobSubmissionHandler

        assert JobSubmissionHandler is not None
    except ImportError:
        pytest.skip("job_submission dependencies not available")


def test_admin_api_handlers_import():
    """Test admin API handlers import."""
    try:
        from services_python.admin_api import (
            CreditsHandler,
            HealthHandler,
            JobsHandler,
            NodesHandler,
        )

        assert CreditsHandler is not None
        assert HealthHandler is not None
        assert JobsHandler is not None
        assert NodesHandler is not None
    except ImportError:
        pytest.skip("admin_api dependencies not available")


def test_voting_system_standalone():
    """Test voting system works standalone."""
    from worker.src.daemon.voting_system import VoteType, VotingSystem

    system = VotingSystem(signing_key=b"test-key-for-voting")
    assert system is not None

    # Create account
    account = system.create_account("test-user")
    assert account.contributor_id == "test-user"
    assert account.balance == 0.0

    # Add credits
    system.add_credits("test-user", 100.0)
    assert account.balance == 100.0

    # Create vote
    vote_id = system.create_vote(
        vote_type=VoteType.JOB_PRIORITY,
        title="Test Vote",
        description="Test description",
        proposer="test-user",
        options=["option1", "option2"],
    )
    assert vote_id is not None
    assert isinstance(vote_id, str)


def test_sandbox_config_validation():
    """Test sandbox config validation."""
    from worker.src.sandbox import SandboxConfig, SandboxType

    # Test defaults
    config = SandboxConfig()
    assert config.max_memory_mb == 4096
    assert config.max_cpu_time_sec == 600
    assert config.sandbox_type == SandboxType.SUBPROCESS
    assert config.network_allowed is True

    # Test custom values
    config2 = SandboxConfig(
        max_memory_mb=8192,
        max_cpu_time_sec=300,
        network_allowed=False,
    )
    assert config2.max_memory_mb == 8192
    assert config2.max_cpu_time_sec == 300
    assert config2.network_allowed is False


def test_credit_ledger_basic():
    """Test credit ledger basic operations."""
    from worker.src.daemon.credit_ledger import CreditEntry, CreditLedger

    ledger = CreditLedger(signing_key=b"test-key-123")
    assert ledger is not None

    # Add credit entry
    entry = CreditEntry(
        contributor_id="node-123",
        job_id="job-456",
        amount=100.0,
        timestamp=1234567890.0,
    )
    assert entry.contributor_id == "node-123"
    assert entry.amount == 100.0

    # Test ledger methods exist
    assert hasattr(ledger, "add_credit")
    assert hasattr(ledger, "get_signed_head")
    assert hasattr(ledger, "verify_chain_integrity")
    assert hasattr(ledger, "get_balance")


def test_benchmark_runner_structure():
    """Test benchmark runner structure."""
    try:
        from worker.src.benchmark import bench_runner

        assert hasattr(bench_runner, "emit")
        assert hasattr(bench_runner, "run_benchmark")
        assert hasattr(bench_runner, "compute_overall")
        assert hasattr(bench_runner, "assign_tier")
        assert hasattr(bench_runner, "main")
    except ImportError:
        pytest.skip("bench_runner not available")
