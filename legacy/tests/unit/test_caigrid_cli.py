"""Tests for DistribAI CLI module."""

import pytest

try:
    from scripts.cli.distribai_cli import (
        Colors,
        HealthChecker,
        JobManager,
        NodeManager,
    )

    HAS_DISTRIBAI_CLI = True
except ImportError:
    HAS_DISTRIBAI_CLI = False
    Colors = None
    NodeManager = None
    JobManager = None
    HealthChecker = None


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_colors_class():
    """Test Colors class exists with expected attributes."""
    assert Colors is not None
    assert hasattr(Colors, "GREEN")
    assert hasattr(Colors, "YELLOW")
    assert hasattr(Colors, "RED")


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_node_manager_import():
    """Test NodeManager imports."""
    assert NodeManager is not None


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_node_manager_creation():
    """Test NodeManager can be instantiated."""
    manager = NodeManager()
    assert manager is not None


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_job_manager_import():
    """Test JobManager imports."""
    assert JobManager is not None


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_job_manager_creation():
    """Test JobManager can be instantiated."""
    manager = JobManager()
    assert manager is not None


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_health_checker_import():
    """Test HealthChecker imports."""
    assert HealthChecker is not None


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_health_checker_creation():
    """Test HealthChecker can be created."""
    checker = HealthChecker()
    assert checker is not None


@pytest.mark.skipif(not HAS_DISTRIBAI_CLI, reason="distribai_cli not available")
def test_health_checker_full_check():
    """Test full health check runs."""
    from unittest import mock

    checker = HealthChecker()
    with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
        result = checker.full_check()
    assert isinstance(result, bool)
