"""
Cross-platform compatibility utilities for DistribAI.

Provides platform-specific optimizations and fallbacks to ensure the system
works reliably across Windows, Linux, macOS, and other platforms.
"""

import os
import platform
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any


class Platform(Enum):
    """Supported platforms."""

    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"


class Architecture(Enum):
    """Supported architectures."""

    X86_64 = "x86_64"
    ARM64 = "arm64"
    UNKNOWN = "unknown"


def get_platform() -> Platform:
    """Get the current platform."""
    system = platform.system().lower()

    if system == "windows":
        return Platform.WINDOWS
    elif system == "linux":
        return Platform.LINUX
    elif system == "darwin":
        return Platform.MACOS
    else:
        return Platform.UNKNOWN


def get_architecture() -> Architecture:
    """Get the current architecture."""
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        return Architecture.X86_64
    elif machine in ("arm64", "aarch64"):
        return Architecture.ARM64
    else:
        return Architecture.UNKNOWN


def is_windows() -> bool:
    """Check if running on Windows."""
    return get_platform() == Platform.WINDOWS


def is_linux() -> bool:
    """Check if running on Linux."""
    return get_platform() == Platform.LINUX


def is_macos() -> bool:
    """Check if running on macOS."""
    return get_platform() == Platform.MACOS


def get_temp_directory() -> Path:
    """Get platform-appropriate temporary directory."""
    if is_windows():
        # On Windows, use %TEMP% or fallback to user temp
        temp_dir = Path(os.environ.get("TEMP", tempfile.gettempdir()))
    else:
        # On Unix-like systems, use /tmp or fallback
        temp_dir = Path("/tmp") if Path("/tmp").exists() else Path(tempfile.gettempdir())

    # Ensure directory exists
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_user_data_directory() -> Path:
    """Get platform-appropriate user data directory."""
    if is_windows():
        data_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif is_macos():
        data_dir = Path.home() / "Library" / "Application Support"
    else:  # Linux and others
        data_dir = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    # Add distribai subdirectory
    distribai_dir = data_dir / "distribai"
    distribai_dir.mkdir(parents=True, exist_ok=True)
    return distribai_dir


def get_user_config_directory() -> Path:
    """Get platform-appropriate user config directory."""
    if is_windows():
        config_dir = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif is_macos():
        config_dir = Path.home() / "Library" / "Preferences"
    else:  # Linux and others
        config_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    # Add distribai subdirectory
    distribai_dir = config_dir / "distribai"
    distribai_dir.mkdir(parents=True, exist_ok=True)
    return distribai_dir


def get_platform_specific_env_vars() -> dict[str, str]:
    """Get platform-specific environment variables."""
    env_vars = {}

    if is_windows():
        env_vars.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONLEGACYWINDOWSSTDIO": "1",
            }
        )
    elif is_macos():
        env_vars.update(
            {
                "OBJC_DISABLE_DEPRECATED": "YES",
                "PYTHONOPTIMIZE": "1",
            }
        )
    elif is_linux():
        env_vars.update(
            {
                "PYTHONUNBUFFERED": "1",
            }
        )

    return env_vars


def get_platform_specific_paths() -> dict[str, str]:
    """Get platform-specific path configurations."""
    paths = {}

    if is_windows():
        paths.update(
            {
                "socket_dir": str(Path("\\\\.\\pipe")),
                "shm_dir": str(Path(tempfile.gettempdir())),
            }
        )
    else:  # Unix-like systems
        paths.update(
            {
                "socket_dir": str(Path("/tmp")),
                "shm_dir": str(Path("/dev/shm"))
                if Path("/dev/shm").exists()
                else str(Path("/tmp")),
            }
        )

    return paths


def safe_import(module_name: str, fallback: Any | None = None) -> Any:
    """Safely import a module with platform-specific fallbacks."""
    try:
        return __import__(module_name)
    except ImportError as e:
        if fallback is not None:
            return fallback

        # Platform-specific import handling
        if module_name == "wmi" and not is_windows():
            # WMI is Windows-only
            return None

        if module_name == "psutil":
            # psutil should be available on all platforms, but might need special handling
            try:
                import psutil

                return psutil
            except ImportError:
                raise ImportError(f"psutil is required on {get_platform().value}: {e}") from e

        raise ImportError(f"Failed to import {module_name} on {get_platform().value}: {e}") from e


def get_platform_specific_limits() -> dict[str, Any]:
    """Get platform-specific resource limits."""
    limits = {
        "max_open_files": 1024,
        "max_memory_gb": 4.0,
        "max_processes": 100,
    }

    if is_windows():
        limits.update(
            {
                "max_open_files": 2048,  # Windows has higher default limits
                "max_memory_gb": 8.0,  # Assume more memory on Windows
            }
        )
    elif is_linux():
        limits.update(
            {
                "max_open_files": 4096,  # Linux typically allows more open files
                "max_memory_gb": 16.0,  # Linux servers often have more memory
            }
        )
    elif is_macos():
        limits.update(
            {
                "max_open_files": 256,  # macOS has lower default limits
                "max_memory_gb": 8.0,  # macOS systems typically have decent memory
            }
        )

    return limits


def get_gpu_backend_priority() -> list[str]:
    """Get GPU backend priority based on platform."""
    if is_windows():
        # Windows: CUDA > DirectML > OpenCL > CPU
        return ["cuda", "directml", "opencl", "cpu"]
    elif is_macos():
        # macOS: Metal > OpenCL > CPU
        return ["mps", "opencl", "cpu"]
    else:  # Linux
        # Linux: CUDA > ROCm > OpenCL > CPU
        return ["cuda", "rocm", "opencl", "cpu"]


def get_platform_specific_optimizations() -> dict[str, Any]:
    """Get platform-specific performance optimizations."""
    optimizations = {
        "use_multiprocessing": True,
        "num_workers": 4,
        "batch_size": 32,
        "memory_limit_gb": 4.0,
        "use_shared_memory": True,
    }

    if is_windows():
        optimizations.update(
            {
                "use_multiprocessing": False,  # Windows has issues with multiprocessing
                "num_workers": 1,  # Use single process on Windows
                "batch_size": 64,  # Larger batches on Windows
                "memory_limit_gb": 6.0,  # Windows typically has more memory
                "use_shared_memory": False,  # Windows shared memory is slower
            }
        )
    elif is_macos():
        optimizations.update(
            {
                "use_multiprocessing": True,  # macOS works well with multiprocessing
                "num_workers": 2,  # Conservative on macOS
                "batch_size": 32,  # Standard batch size
                "memory_limit_gb": 8.0,  # macOS systems often have good memory
                "use_shared_memory": True,  # macOS shared memory works well
            }
        )
    else:  # Linux
        optimizations.update(
            {
                "use_multiprocessing": True,  # Linux works best with multiprocessing
                "num_workers": 8,  # Use more workers on Linux
                "batch_size": 128,  # Larger batches on Linux
                "memory_limit_gb": 16.0,  # Linux servers often have more memory
                "use_shared_memory": True,  # Linux shared memory is fastest
            }
        )

    return optimizations


def check_platform_compatibility() -> dict[str, Any]:
    """Check platform compatibility and return status."""
    platform_info = {
        "platform": get_platform().value,
        "architecture": get_architecture().value,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "compatibility": "unknown",
        "warnings": [],
        "recommendations": [],
    }

    # Check Python version

    # Check platform-specific requirements
    if is_windows():
        # Check for Windows-specific issues
        import importlib.util

        if importlib.util.find_spec("win32api") is None:
            platform_info["warnings"].append("pywin32 not available")
            platform_info["recommendations"].append(
                "Install pywin32 for better Windows integration"
            )

    elif is_macos():
        # Check for macOS-specific issues
        macos_version = platform.mac_ver()[0]
        if macos_version and tuple(map(int, macos_version.split("."))) < (10, 15):
            platform_info["warnings"].append("macOS 10.15+ recommended")
            platform_info["recommendations"].append("Upgrade to macOS 10.15 (Catalina) or later")

    elif is_linux():
        # Check for Linux-specific requirements
        import importlib.util

        if importlib.util.find_spec("psutil") is None:
            platform_info["warnings"].append("psutil not available")
            platform_info["recommendations"].append("Install psutil for system monitoring")

    # Determine overall compatibility
    if not platform_info["warnings"]:
        platform_info["compatibility"] = "excellent"
    elif len(platform_info["warnings"]) <= 2:
        platform_info["compatibility"] = "good"
    elif len(platform_info["warnings"]) <= 4:
        platform_info["compatibility"] = "fair"
    else:
        platform_info["compatibility"] = "poor"

    return platform_info


def setup_platform_environment() -> None:
    """Setup platform-specific environment variables and configurations."""
    # Set platform-specific environment variables
    env_vars = get_platform_specific_env_vars()
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    # Create platform-specific directories
    get_user_data_directory()
    get_user_config_directory()

    # Set up platform-specific optimizations
    optimizations = get_platform_specific_optimizations()
    for key, value in optimizations.items():
        os.environ[f"DISTRIBAI_{key.upper()}"] = str(value)


# Global platform information
CURRENT_PLATFORM = get_platform()
CURRENT_ARCH = get_architecture()
PLATFORM_INFO = check_platform_compatibility()

# Setup platform environment on import
setup_platform_environment()
