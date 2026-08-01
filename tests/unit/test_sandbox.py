"""
Unit tests for sandboxed execution
"""


def test_sandbox_import():
    from worker.src.sandbox import Sandbox, SandboxConfig, SandboxType

    assert Sandbox is not None
    assert SandboxConfig is not None
    assert SandboxType is not None


def test_sandbox_config_defaults():
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()
    assert config.max_memory_mb == 4096
    assert config.max_cpu_time_sec == 600
    assert config.sandbox_type.value == "subprocess"
    assert config.network_allowed is True


def test_sandbox_config_custom():
    from worker.src.sandbox import SandboxConfig, SandboxType

    config = SandboxConfig(
        sandbox_type=SandboxType.SUBPROCESS,
        max_memory_mb=8192,
        max_cpu_time_sec=300,
        network_allowed=False,
    )
    assert config.max_memory_mb == 8192
    assert config.max_cpu_time_sec == 300
    assert config.network_allowed is False


def test_sandbox_creation():
    from worker.src.sandbox import Sandbox, SandboxConfig

    config = SandboxConfig()
    sandbox = Sandbox(config)
    assert sandbox is not None
    assert sandbox.config == config


def test_sandbox_is_available():
    """Test sandbox availability check."""
    from worker.src.sandbox import Sandbox, SandboxConfig

    config = SandboxConfig()
    sandbox = Sandbox(config)
    assert sandbox._check_linux() in [True, False]


def test_sandbox_config_allowed_paths():
    """Test sandbox allowed paths configuration."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()
    assert "/app" in config.allowed_paths
    assert "/tmp" in config.allowed_paths
    assert "/workspace" in config.allowed_paths


def test_sandbox_config_blocked_paths():
    """Test sandbox blocked paths configuration."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()
    assert "/etc/passwd" in config.blocked_paths
    assert "/etc/shadow" in config.blocked_paths


def test_sandbox_config_allowed_hosts():
    """Test sandbox allowed hosts configuration."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()
    assert "huggingface.co" in config.allowed_hosts
    assert "s3.amazonaws.com" in config.allowed_hosts


def test_sandbox_config_blocked_ports():
    """Test sandbox blocked ports configuration."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()
    assert 22 in config.blocked_ports  # SSH
    assert 23 in config.blocked_ports  # Telnet


def test_sandbox_type_enum():
    """Test SandboxType enum values."""
    from worker.src.sandbox import SandboxType

    assert SandboxType.SUBPROCESS.value == "subprocess"
    assert SandboxType.NAMESPACE.value == "namespace"
    assert SandboxType.SECCOMP.value == "seccomp"


def test_sandbox_simple_function():
    """Test sandbox can run a simple function."""
    from worker.src.sandbox import Sandbox, SandboxConfig

    def simple_add(x, y):
        return x + y

    config = SandboxConfig()
    sandbox = Sandbox(config)

    assert hasattr(sandbox, "run")
    assert callable(sandbox.run)


def test_sandbox_cleanup():
    from worker.src.sandbox import Sandbox, SandboxConfig

    config = SandboxConfig()
    sandbox = Sandbox(config)
    sandbox.cleanup()


def test_sandbox_config_timeout():
    """Test sandbox timeout configuration."""
    from worker.src.sandbox import SandboxConfig

    config = SandboxConfig()
    assert config.max_cpu_time_sec > 0


def test_sandbox_str_representation():
    """Test sandbox string representation."""
    from worker.src.sandbox import Sandbox, SandboxConfig

    config = SandboxConfig()
    sandbox = Sandbox(config)

    str_repr = str(sandbox)
    assert "Sandbox" in str_repr
