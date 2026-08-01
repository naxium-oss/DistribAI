"""Extended tests for CLI modules."""

import pytest

try:
    from scripts.cli.distribai_cli import (
        Colors,
        HealthChecker,
        JobManager,
        NodeManager,
        print_error,
        print_header,
        print_info,
        print_success,
        print_warning,
    )

    HAS_DISTRIBAI_CLI = True
except ImportError:
    HAS_DISTRIBAI_CLI = False


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_colors_attributes():
    """Test Colors class has all expected color attributes."""
    assert hasattr(Colors, "GREEN")
    assert hasattr(Colors, "YELLOW")
    assert hasattr(Colors, "RED")
    assert hasattr(Colors, "BLUE")
    assert hasattr(Colors, "CYAN")
    assert hasattr(Colors, "BOLD")
    assert hasattr(Colors, "END")


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_print_functions():
    """Test print helper functions exist."""
    assert callable(print_header)
    assert callable(print_success)
    assert callable(print_warning)
    assert callable(print_error)
    assert callable(print_info)


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_node_manager_methods():
    """Test NodeManager has expected methods."""
    manager = NodeManager()

    expected_methods = [
        "get_config",
        "set_config",
        "status",
        "set_resources",
        "set_region",
        "set_name",
    ]

    for method in expected_methods:
        assert hasattr(manager, method), f"Missing method: {method}"


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_job_manager_methods():
    """Test JobManager has expected methods."""
    manager = JobManager()

    expected_methods = [
        "_api_call",
        "list_jobs",
        "create_job",
        "cancel_job",
    ]

    for method in expected_methods:
        assert hasattr(manager, method), f"Missing method: {method}"


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_health_checker_individual_checks():
    """Test HealthChecker individual check methods."""
    checker = HealthChecker()

    # Test full_check method exists
    assert hasattr(checker, "full_check")


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_health_checker_full_check_returns_bool():
    """Test HealthChecker.full_check returns boolean."""
    from unittest import mock

    checker = HealthChecker()
    with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
        result = checker.full_check()
    assert isinstance(result, bool)


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_colors_are_strings():
    """Test that color codes are strings."""
    assert isinstance(Colors.GREEN, str)
    assert isinstance(Colors.RED, str)
    assert isinstance(Colors.END, str)
