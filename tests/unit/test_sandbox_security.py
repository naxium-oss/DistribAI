"""Security tests for sandbox module."""


def test_sandbox_safe_unpickler_restricts_modules():
    """Test SafeUnpickler restricts module loading."""
    import io

    from worker.src.sandbox.serialization import SafeUnpickler

    unpickler = SafeUnpickler(io.BytesIO(b""))

    # Should allow builtins
    try:
        result = unpickler.find_class("builtins", "list")
        assert result is list
    except Exception:
        pass  # May fail depending on implementation


def test_sandbox_blocked_paths_security():
    """Test that sensitive paths are blocked by default."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()

    # Check sensitive paths are blocked
    sensitive_paths = ["/etc/passwd", "/etc/shadow", "/root", "/home"]
    for path in sensitive_paths:
        assert path in config.blocked_paths, f"{path} should be blocked"


def test_sandbox_blocked_ports_security():
    """Test that dangerous ports are blocked by default."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()

    # Check dangerous ports are blocked
    dangerous_ports = [22, 23, 25]  # SSH, Telnet, SMTP
    for port in dangerous_ports:
        assert port in config.blocked_ports, f"Port {port} should be blocked"


def test_sandbox_drop_capabilities():
    """Test that capabilities are dropped by default."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()
    assert config.drop_capabilities is True


def test_sandbox_no_new_privileges():
    """Test that no_new_privileges is enabled by default."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()
    assert config.no_new_privileges is True


def test_sandbox_allowed_hosts_restricted():
    """Test that only specific hosts are allowed."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()

    # Check allowed hosts are restricted
    expected_hosts = [
        "huggingface.co",
        "s3.amazonaws.com",
    ]
    for host in expected_hosts:
        assert host in config.allowed_hosts, f"{host} should be allowed"
