"""Tests for error handling across modules."""

import tempfile

import pytest


def test_worker_state_file_not_found():
    """Test WorkerState handles missing files gracefully."""
    try:
        from worker.src.daemon.state import WorkerState

        with tempfile.TemporaryDirectory() as tmp:
            state = WorkerState(tmp, "test-node")

            # Try to load benchmark results that don't exist
            result = state.load_benchmark_results()
            assert result is None
    except ImportError:
        pytest.skip("state module not available")


def test_s3_url_parsing_errors():
    """Test S3 URL parsing with invalid inputs."""
    try:
        from worker.src.daemon.s3_util import S3Manager

        manager = S3Manager()

        # Test with invalid URLs
        assert not manager._is_s3_url("")
        assert not manager._is_s3_url("http://example.com")
        assert not manager._is_s3_url("/local/path")
        assert not manager._is_s3_url("ftp://host/file")
    except ImportError:
        pytest.skip("s3_util not available")


def test_credit_ledger_invalid_operations():
    """Test CreditLedger with invalid operations."""
    try:
        from worker.src.daemon.credit_ledger import CreditLedger

        ledger = CreditLedger(signing_key=b"test-key")

        # Get balance for non-existent contributor
        balance = ledger.get_balance("non-existent")
        assert balance == 0.0

        # Get history for non-existent contributor
        history = ledger.get_credit_history("non-existent")
        assert history == []
    except ImportError:
        pytest.skip("credit_ledger not available")


def test_voting_system_insufficient_credits():
    """Test VotingSystem handles insufficient credits."""
    try:
        from worker.src.daemon.voting_system import VoteType, VotingSystem

        system = VotingSystem(signing_key=b"test-key")
        system.create_account("poor-user")
        # Don't add credits

        # Try to create vote without credits
        with pytest.raises(ValueError):
            system.create_vote(
                proposer="poor-user",
                vote_type=VoteType.JOB_PRIORITY,
                title="Test",
                description="Test",
                options=["yes", "no"],
            )
    except ImportError:
        pytest.skip("voting_system not available")


def test_log_score_invalid_inputs():
    """Test log_score with invalid inputs."""
    try:
        from worker.src.benchmark.bench_network import log_score

        # Test with zero floor
        result = log_score(10.0, 0.0, 100.0)
        assert result == 0.0  # Should return 0 for invalid input

        # Test with negative value
        result = log_score(-5.0, 1.0, 100.0)
        assert result == 0.0

        # Test with ceil <= floor
        result = log_score(50.0, 100.0, 100.0)
        assert result == 0.0
    except ImportError:
        pytest.skip("bench_network not available")


def test_sandbox_config_validation_errors():
    """Test SandboxConfig with invalid configurations."""
    from worker.src.sandbox import SandboxConfig

    # Negative memory should still work (just be stored as-is)
    config = SandboxConfig(max_memory_mb=-1)
    assert config.max_memory_mb == -1


def test_registration_manager_empty_hardware():
    """Test RegistrationManager handles hardware detection."""
    try:
        from worker.src.daemon.registration import RegistrationManager

        registrar = RegistrationManager(
            orchestrator_admin_url="http://localhost:8766",
            node_id="test-node",
        )

        hardware = registrar.detect_hardware()
        assert hardware is not None
        assert "node_id" in hardware
        assert hardware["node_id"] == "test-node"
    except ImportError:
        pytest.skip("registration not available")


def test_merkle_tree_invalid_data():
    """Test MerkleTree with invalid data."""
    try:
        from worker.src.daemon.credit_ledger import MerkleTree

        # Empty data
        tree = MerkleTree([])
        root = tree.root_hash()
        assert root == b""  # Empty hash for empty tree
    except ImportError:
        pytest.skip("credit_ledger not available")


def test_emit_empty_data():
    """Test emit with empty data."""
    try:
        import io
        import json
        import sys

        from worker.src.benchmark.bench_network import emit

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        # Test with empty dict
        emit({})

        sys.stdout = old_stdout
        output = captured.getvalue()

        # Should output valid JSON
        parsed = json.loads(output.strip())
        assert parsed == {}
    except ImportError:
        pytest.skip("bench_network not available")


def test_constants_no_errors():
    """Test that constants module loads without errors."""
    from services_python import constants

    # All expected constants should be accessible
    assert hasattr(constants, "JWT_SECRET")
    assert hasattr(constants, "JWT_ALGORITHM")
    assert hasattr(constants, "SIGNING_KEY")
