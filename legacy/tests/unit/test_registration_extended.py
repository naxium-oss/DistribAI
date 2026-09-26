"""Extended tests for registration module to increase coverage."""

import pytest

try:
    from worker.src.daemon.registration import (
        RedactingFilter,
        RegistrationManager,
        _sanitize_log_message,
    )

    HAS_REGISTRATION = True
except ImportError:
    HAS_REGISTRATION = False


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_sanitize_log_message_basic():
    """Test _sanitize_log_message basic functionality."""
    result = _sanitize_log_message("Normal message")
    assert result == "Normal message"


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_sanitize_log_message_jwt():
    """Test _sanitize_log_message redacts JWT."""
    msg_with_jwt = "Token: eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.test"
    result = _sanitize_log_message(msg_with_jwt)
    assert "[REDACTED" in result


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_sanitize_log_message_auth():
    """Test _sanitize_log_message redacts Authorization."""
    msg = "Authorization: Bearer secret-token"
    result = _sanitize_log_message(msg)
    assert "[REDACTED]" in result


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_sanitize_log_message_password():
    """Test _sanitize_log_message redacts password."""
    msg = "password=secret123"
    result = _sanitize_log_message(msg)
    assert "[REDACTED]" in result


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_sanitize_log_message_empty():
    """Test _sanitize_log_message with empty string."""
    result = _sanitize_log_message("")
    assert result == ""


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_redacting_filter():
    """Test RedactingFilter processes log records."""
    import logging

    filter_obj = RedactingFilter()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    result = filter_obj.filter(record)
    assert result is True


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_redacting_filter_with_sensitive():
    """Test RedactingFilter redacts sensitive data."""
    import logging

    filter_obj = RedactingFilter()

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="password=secret123",
        args=(),
        exc_info=None,
    )

    filter_obj.filter(record)
    assert "[REDACTED]" in record.msg


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_registration_manager_init():
    """Test RegistrationManager initialization."""
    registrar = RegistrationManager(
        orchestrator_admin_url="http://localhost:8766",
        node_id="test-node",
    )

    assert registrar.admin_url == "http://localhost:8766"
    assert registrar.node_id == "test-node"


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_registration_manager_detect_hardware():
    """Test RegistrationManager detect_hardware."""
    registrar = RegistrationManager(
        orchestrator_admin_url="http://localhost:8766",
        node_id="test-node",
    )

    hardware = registrar.detect_hardware()

    assert isinstance(hardware, dict)
    assert "node_id" in hardware
    assert hardware["node_id"] == "test-node"
    assert "os" in hardware
    assert "cpu_count" in hardware
    assert "ram_total_gb" in hardware
    assert "ts" in hardware


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_registration_manager_detect_hardware_types():
    """Test detect_hardware returns correct types."""
    registrar = RegistrationManager(
        orchestrator_admin_url="http://localhost:8766",
        node_id="test-node",
    )

    hardware = registrar.detect_hardware()

    assert isinstance(hardware["node_id"], str)
    assert isinstance(hardware["os"], str)
    assert isinstance(hardware["ram_total_gb"], float)
    assert isinstance(hardware["ts"], int)


@pytest.mark.skipif(not HAS_REGISTRATION, reason="registration not available")
def test_registration_manager_detect_hardware_gpu_fields():
    """Test detect_hardware includes GPU fields."""
    registrar = RegistrationManager(
        orchestrator_admin_url="http://localhost:8766",
        node_id="test-node",
    )

    hardware = registrar.detect_hardware()

    # GPU fields should exist (may be None or "CPU-Only")
    assert "gpu_model" in hardware
    assert "vram_mb" in hardware
    # cuda_version and driver_version may not be present on CPU-only systems
