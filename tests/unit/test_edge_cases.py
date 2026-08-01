"""Edge case and error handling tests."""

import tempfile

import pytest


def test_worker_state_with_invalid_paths():
    """Test WorkerState handles invalid paths gracefully."""
    try:
        from worker.src.daemon.state import WorkerState

        with tempfile.TemporaryDirectory() as tmp:
            state = WorkerState(tmp, "test-node")
            # Should handle invalid path gracefully
            result = state.load_benchmark_results()
            assert result is None  # No results file exists
    except ImportError:
        pytest.skip("state module not available")


def test_sandbox_config_edge_cases():
    """Test SandboxConfig with edge case values."""
    from worker.src.sandbox import SandboxConfig

    # Test with zero values
    config = SandboxConfig(max_memory_mb=0)
    assert config.max_memory_mb == 0

    # Test with very large values
    config2 = SandboxConfig(max_memory_mb=1000000)
    assert config2.max_memory_mb == 1000000


def test_credit_ledger_empty_operations():
    """Test CreditLedger with empty operations."""
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


def test_voting_system_edge_cases():
    """Test VotingSystem edge cases."""
    try:
        from worker.src.daemon.voting_system import VotingSystem

        system = VotingSystem(signing_key=b"test-key")

        # Create account with empty ID
        account = system.create_account("")
        assert account.contributor_id == ""

        # Add zero credits
        system.add_credits("test", 0.0)
        # Should not crash
    except ImportError:
        pytest.skip("voting_system not available")


def test_log_score_edge_cases():
    """Test log_score with edge case values."""
    try:
        from worker.src.benchmark.bench_network import log_score

        # Test with negative values
        result = log_score(-10.0, 1.0, 100.0)
        assert result == 0.0

        # Test with very small values
        result = log_score(0.0001, 0.00001, 1.0)
        assert 0.0 <= result <= 100.0

        # Test with infinity-like values
        result = log_score(1e10, 1.0, 100.0)
        assert result == 100.0
    except ImportError:
        pytest.skip("bench_network not available")


def test_emit_with_special_characters():
    """Test emit with special characters in data."""
    try:
        import io
        import json
        import sys

        from worker.src.benchmark.bench_network import emit

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        # Test with unicode and special chars
        data = {"message": "Special chars: äöü ñ 中文 🎉", "value": 42}
        emit(data)

        sys.stdout = old_stdout
        output = captured.getvalue()

        # Should be valid JSON
        parsed = json.loads(output.strip())
        assert parsed["value"] == 42
    except ImportError:
        pytest.skip("bench_network not available")


def test_compute_backend_unavailable():
    """Test compute backend when dependencies unavailable."""
    try:
        from worker.src.compute import detect_backend

        # Should return None or CPUBackend (which is always available)
        _backend = detect_backend()  # noqa: F841
        # Should not crash
    except ImportError:
        pytest.skip("compute module not available")


def test_s3_url_parsing_edge_cases():
    """Test S3 URL parsing with edge cases."""
    try:
        from worker.src.daemon.s3_util import S3Manager

        manager = S3Manager()

        # Test with non-S3 URL
        assert not manager._is_s3_url("http://example.com")
        assert not manager._is_s3_url("/local/path")
        assert not manager._is_s3_url("")
    except ImportError:
        pytest.skip("s3_util not available")


def test_constants_types():
    """Test that constants have correct types."""
    from services_python import constants

    # Check JWT constants
    assert isinstance(constants.JWT_SECRET, str)
    assert isinstance(constants.JWT_ALGORITHM, str)
    assert len(constants.JWT_SECRET) > 0
    assert constants.JWT_ALGORITHM == "HS256"


def test_merkle_tree_empty():
    """Test MerkleTree with empty data."""
    try:
        from worker.src.daemon.credit_ledger import MerkleTree

        tree = MerkleTree([])
        root = tree.root_hash()
        assert root == b""  # Empty tree returns empty hash
    except ImportError:
        pytest.skip("credit_ledger not available")


def test_stake_info_defaults():
    """Test StakeInfo default values."""
    try:
        from worker.src.daemon.credit_ledger import StakeInfo

        stake = StakeInfo(
            contributor_id="test",
            amount=100.0,
            staked_at=1234567890.0,
            lock_period_days=30,
            unlocks_at=1234567890.0,
            purpose="test",
        )
        assert stake.slashed_amount == 0.0  # Default value
    except ImportError:
        pytest.skip("StakeInfo not available")
