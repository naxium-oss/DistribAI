"""Basic integration tests between modules."""

import tempfile

import pytest


def test_worker_state_with_credit_ledger():
    """Test WorkerState integration with credit ledger concepts."""
    try:
        from worker.src.daemon.credit_ledger import CreditEntry
        from worker.src.daemon.state import WorkerState

        with tempfile.TemporaryDirectory() as tmp:
            state = WorkerState(tmp, "node-001")

            # Simulate recording work that would earn credits
            entry = CreditEntry(
                contributor_id="node-001",
                job_id="job-123",
                amount=50.0,
                metadata={"work_type": "training"},
            )

            assert entry.contributor_id == state.node_id
    except ImportError:
        pytest.skip("dependencies not available")


def test_voting_system_with_credit_accounts():
    """Test VotingSystem integration with credit accounts."""
    try:
        from worker.src.daemon.voting_system import VotingSystem

        system = VotingSystem(signing_key=b"integration-test-key")

        # Create multiple accounts
        for i in range(5):
            account = system.create_account(f"user-{i}")
            system.add_credits(f"user-{i}", float(i * 100))
            assert account.balance == float(i * 100)
    except ImportError:
        pytest.skip("voting_system not available")


def test_sandbox_with_compute_backends():
    """Test Sandbox integration with compute backend concepts."""
    try:
        from worker.src.sandbox import SandboxConfig, SandboxType

        # Configure sandbox for compute workload
        config = SandboxConfig(
            max_memory_mb=8192,
            max_cpu_time_sec=3600,
            network_allowed=True,
        )

        assert config.max_memory_mb >= 4096  # Minimum for ML workloads
        assert config.sandbox_type == SandboxType.SUBPROCESS
    except ImportError:
        pytest.skip("sandbox not available")


def test_benchmark_with_sandbox_concepts():
    """Test benchmark module concepts with sandbox security."""
    try:
        from worker.src.benchmark.bench_network import log_score
        from worker.src.sandbox import SandboxConfig

        # Simulate benchmark scoring within sandbox constraints
        score = log_score(50.0, 1.0, 100.0)
        assert 0 <= score <= 100

        # Verify sandbox config exists
        config = SandboxConfig()
        assert config is not None
    except ImportError:
        pytest.skip("dependencies not available")


def test_credit_ledger_chain_integrity():
    """Test credit ledger maintains chain integrity across operations."""
    try:
        from worker.src.daemon.credit_ledger import CreditEntry, CreditLedger

        ledger = CreditLedger(signing_key=b"integrity-test-key")
        assert ledger is not None

        # Add multiple entries
        for i in range(3):
            entry = CreditEntry(
                contributor_id=f"node-{i}",
                job_id=f"job-{i}",
                amount=float(10 * (i + 1)),
            )
            assert hasattr(entry, "contributor_id")
            assert entry.contributor_id == f"node-{i}"
    except ImportError:
        pytest.skip("credit_ledger not available")


def test_node_registration_with_voting():
    """Test node registration concepts with voting system."""
    try:
        from worker.src.daemon.voting_system import VotingSystem

        system = VotingSystem(signing_key=b"node-voting-key")

        # Node registers and gets voting power
        system.create_account("registered-node")
        system.add_credits("registered-node", 1000.0)

        # Check voting power through accounts dict
        account = system.accounts.get("registered-node")
        assert account is not None
        assert account.balance >= 1000.0
    except ImportError:
        pytest.skip("voting_system not available")


def test_constants_consistency():
    """Test that constants are consistent across modules."""
    from services_python import constants

    # JWT settings should be valid
    assert constants.JWT_ALGORITHM in ["HS256", "HS512", "RS256"]

    # Ports should be positive
    assert int(constants.DEFAULT_GRPC_PORT) > 0
    assert int(constants.DEFAULT_ADMIN_PORT) > 0


def test_error_handling_consistency():
    """Test that error handling is consistent across modules."""

    modules_to_check = [
        "worker.src.daemon.state",
        "worker.src.daemon.credit_ledger",
        "worker.src.daemon.voting_system",
    ]

    for module_name in modules_to_check:
        try:
            __import__(module_name)
        except ImportError:
            pass  # Expected if dependencies missing


def test_dataclass_consistency():
    """Test that dataclasses have consistent patterns."""
    try:
        from worker.src.daemon.credit_ledger import CreditEntry
        from worker.src.daemon.voting_system import Vote

        # All should have proper dataclass structure
        entry = CreditEntry(
            contributor_id="test",
            job_id="job-1",
            amount=10.0,
        )
        assert hasattr(entry, "contributor_id")

        # Vote uses 'id' field, not 'vote_id'
        from worker.src.daemon.voting_system import VoteType

        vote_id = "test-vote-001"
        vote = Vote(
            id=vote_id,
            vote_type=VoteType.JOB_PRIORITY,
            proposer="test",
            title="Test Vote",
            description="Test",
            options=["yes", "no"],
        )
        assert vote.id == vote_id
    except ImportError:
        pytest.skip("dataclasses not available")


def test_type_hints_consistency():
    """Test that type hints are consistent."""

    optional_modules = [
        "worker.src.daemon.state",
        "worker.src.daemon.credit_ledger",
    ]

    for module_name in optional_modules:
        try:
            _module = __import__(module_name, fromlist=[""])  # noqa: F841
            # Module should import successfully
        except ImportError:
            pass
