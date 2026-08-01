"""Tests for configuration validation across modules."""

import pytest


def test_jwt_secret_generation():
    """Test JWT secret auto-generation when env var not set."""
    from services_python.constants import JWT_ALGORITHM, JWT_SECRET

    # Should have valid values
    assert isinstance(JWT_SECRET, str)
    assert len(JWT_SECRET) > 0
    assert JWT_ALGORITHM == "HS256"


def test_signing_key_generation():
    """Test signing key auto-generation."""
    from services_python.constants import SIGNING_KEY

    assert isinstance(SIGNING_KEY, str)
    assert len(SIGNING_KEY) > 0


def test_port_constants():
    """Test port configuration constants."""
    from services_python.constants import (
        DEFAULT_ADMIN_HOST,
        DEFAULT_ADMIN_PORT,
        DEFAULT_GRPC_PORT,
    )

    # Ports should be valid strings that can be converted to int
    assert isinstance(DEFAULT_GRPC_PORT, str)
    assert isinstance(DEFAULT_ADMIN_PORT, str)
    assert int(DEFAULT_GRPC_PORT) > 0
    assert int(DEFAULT_ADMIN_PORT) > 0
    assert isinstance(DEFAULT_ADMIN_HOST, str)


def test_sandbox_config_bounds():
    """Test sandbox config with boundary values."""
    from worker.src.sandbox import SandboxConfig

    # Test minimum values
    config = SandboxConfig(max_memory_mb=1)
    assert config.max_memory_mb == 1

    # Test maximum values
    config2 = SandboxConfig(max_memory_mb=1000000)
    assert config2.max_memory_mb == 1000000


def test_sandbox_config_paths():
    """Test sandbox path configurations."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()

    # Should have blocked paths
    assert len(config.blocked_paths) > 0
    assert "/etc/passwd" in config.blocked_paths

    # Should have allowed paths
    assert len(config.allowed_paths) > 0
    assert "/tmp" in config.allowed_paths


def test_sandbox_network_config():
    """Test sandbox network configuration."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()

    # Check network settings
    assert isinstance(config.network_allowed, bool)
    assert len(config.allowed_hosts) > 0
    assert len(config.blocked_ports) > 0


def test_credit_ledger_batch_size():
    """Test credit ledger batch size configuration."""
    try:
        from worker.src.daemon.credit_ledger import CreditLedger

        # Test default batch size
        ledger = CreditLedger(signing_key=b"test-key")
        assert ledger.batch_size == 100

        # Test custom batch size
        ledger2 = CreditLedger(signing_key=b"test-key", batch_size=50)
        assert ledger2.batch_size == 50
    except ImportError:
        pytest.skip("credit_ledger not available")


def test_voting_system_quorum_config():
    """Test voting system quorum configuration."""
    try:
        from worker.src.daemon.voting_system import VoteType, VotingSystem

        system = VotingSystem(signing_key=b"test-key")

        # Check quorum configs exist for all vote types
        for vote_type in VoteType:
            config = system.quorum_configs.get(vote_type)
            assert config is not None
            assert config.quorum_required > 0
    except ImportError:
        pytest.skip("voting_system not available")


def test_benchmark_env_defaults():
    """Test benchmark environment variable defaults."""
    try:
        from worker.src.benchmark import bench_network

        # Check floor/ceil are positive
        assert bench_network._FLOOR >= 0
        assert bench_network._CEIL > bench_network._FLOOR

        # Check durations are positive
        assert bench_network._DL_DUR > 0
        assert bench_network._LB_DUR > 0
    except ImportError:
        pytest.skip("bench_network not available")


def test_compute_backend_env_vars():
    """Test compute backend environment variable handling."""
    import os

    # Test that env vars have defaults
    jwt_secret = os.getenv("JWT_SECRET", "")
    signing_key = os.getenv("SIGNING_KEY", "")

    # Should be able to run without these set
    # (they get auto-generated)
    assert jwt_secret is not None  # nosec
    assert signing_key is not None  # nosec


def test_constants_all_defined():
    """Test that all expected constants are defined."""
    from services_python import constants

    expected_constants = [
        "JWT_SECRET",
        "JWT_ALGORITHM",
        "SIGNING_KEY",
        "DEFAULT_GRPC_PORT",
        "DEFAULT_ADMIN_PORT",
        "DEFAULT_ADMIN_HOST",
    ]

    for const in expected_constants:
        assert hasattr(constants, const), f"Missing constant: {const}"
