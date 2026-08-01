"""Tests for cross-platform compatibility utilities."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the module under test
import services_python.platform_utils as platform_utils


class TestPlatformDetection:
    """Tests for platform detection functions."""

    def test_get_platform(self):
        """Test platform detection."""
        result = platform_utils.get_platform()
        assert result in platform_utils.Platform
        assert isinstance(result.value, str)

    def test_get_architecture(self):
        """Test architecture detection."""
        result = platform_utils.get_architecture()
        assert result in platform_utils.Architecture
        assert isinstance(result.value, str)

    def test_is_windows(self):
        """Test Windows detection."""
        result = platform_utils.is_windows()
        assert isinstance(result, bool)

    def test_is_linux(self):
        """Test Linux detection."""
        result = platform_utils.is_linux()
        assert isinstance(result, bool)

    def test_is_macos(self):
        """Test macOS detection."""
        result = platform_utils.is_macos()
        assert isinstance(result, bool)

    @patch("platform.system")
    def test_get_platform_windows(self, mock_system):
        """Test Windows platform detection."""
        mock_system.return_value = "Windows"
        result = platform_utils.get_platform()
        assert result == platform_utils.Platform.WINDOWS

    @patch("platform.system")
    def test_get_platform_linux(self, mock_system):
        """Test Linux platform detection."""
        mock_system.return_value = "Linux"
        result = platform_utils.get_platform()
        assert result == platform_utils.Platform.LINUX

    @patch("platform.system")
    def test_get_platform_macos(self, mock_system):
        """Test macOS platform detection."""
        mock_system.return_value = "Darwin"
        result = platform_utils.get_platform()
        assert result == platform_utils.Platform.MACOS


class TestPlatformPaths:
    """Tests for platform-specific path utilities."""

    def test_get_temp_directory(self):
        """Test getting temporary directory."""
        result = platform_utils.get_temp_directory()
        assert isinstance(result, Path)
        assert result.exists()

    def test_get_user_data_directory(self):
        """Test getting user data directory."""
        result = platform_utils.get_user_data_directory()
        assert isinstance(result, Path)
        assert result.exists()
        assert result.name == "distribai"

    def test_get_user_config_directory(self):
        """Test getting user config directory."""
        result = platform_utils.get_user_config_directory()
        assert isinstance(result, Path)
        assert result.exists()
        assert result.name == "distribai"

    @patch("services_python.platform_utils.is_windows")
    def test_get_platform_specific_paths_windows(self, mock_is_windows):
        """Test getting platform-specific paths on Windows."""
        mock_is_windows.return_value = True
        result = platform_utils.get_platform_specific_paths()
        assert "socket_dir" in result
        assert "shm_dir" in result

    @patch("services_python.platform_utils.is_windows")
    def test_get_platform_specific_paths_unix(self, mock_is_windows):
        """Test getting platform-specific paths on Unix-like systems."""
        mock_is_windows.return_value = False
        result = platform_utils.get_platform_specific_paths()
        assert "socket_dir" in result
        assert "shm_dir" in result


class TestPlatformEnvironment:
    """Tests for platform-specific environment utilities."""

    def test_get_platform_specific_env_vars(self):
        """Test getting platform-specific environment variables."""
        result = platform_utils.get_platform_specific_env_vars()
        assert isinstance(result, dict)
        assert len(result) > 0

    @patch("services_python.platform_utils.is_windows")
    def test_get_platform_specific_env_vars_windows(self, mock_is_windows):
        """Test getting Windows-specific environment variables."""
        mock_is_windows.return_value = True
        result = platform_utils.get_platform_specific_env_vars()
        assert "PYTHONIOENCODING" in result
        assert "PYTHONLEGACYWINDOWSSTDIO" in result

    @patch("services_python.platform_utils.is_macos")
    def test_get_platform_specific_env_vars_macos(self, mock_is_macos):
        """Test getting macOS-specific environment variables."""
        mock_is_macos.return_value = True
        with (
            patch("services_python.platform_utils.is_windows", return_value=False),
            patch("services_python.platform_utils.is_linux", return_value=False),
        ):
            result = platform_utils.get_platform_specific_env_vars()
            assert "OBJC_DISABLE_DEPRECATED" in result

    @patch("services_python.platform_utils.is_linux")
    def test_get_platform_specific_env_vars_linux(self, mock_is_linux):
        """Test getting Linux-specific environment variables."""
        mock_is_linux.return_value = True
        with (
            patch("services_python.platform_utils.is_windows", return_value=False),
            patch("services_python.platform_utils.is_macos", return_value=False),
        ):
            result = platform_utils.get_platform_specific_env_vars()
            assert "PYTHONUNBUFFERED" in result


class TestPlatformLimits:
    """Tests for platform-specific resource limits."""

    def test_get_platform_specific_limits(self):
        """Test getting platform-specific resource limits."""
        result = platform_utils.get_platform_specific_limits()
        assert isinstance(result, dict)
        assert "max_open_files" in result
        assert "max_memory_gb" in result
        assert "max_processes" in result

    @patch("services_python.platform_utils.is_windows")
    def test_get_platform_specific_limits_windows(self, mock_is_windows):
        """Test getting Windows-specific resource limits."""
        mock_is_windows.return_value = True
        result = platform_utils.get_platform_specific_limits()
        assert result["max_open_files"] >= 2048
        assert result["max_memory_gb"] >= 8.0

    @patch("services_python.platform_utils.is_linux")
    def test_get_platform_specific_limits_linux(self, mock_is_linux):
        """Test getting Linux-specific resource limits."""
        mock_is_linux.return_value = True
        with (
            patch("services_python.platform_utils.is_windows", return_value=False),
            patch("services_python.platform_utils.is_macos", return_value=False),
        ):
            result = platform_utils.get_platform_specific_limits()
            assert result["max_open_files"] >= 4096
            assert result["max_memory_gb"] >= 16.0

    @patch("services_python.platform_utils.is_macos")
    def test_get_platform_specific_limits_macos(self, mock_is_macos):
        """Test getting macOS-specific resource limits."""
        mock_is_macos.return_value = True
        result = platform_utils.get_platform_specific_limits()
        assert result["max_open_files"] >= 256
        assert result["max_memory_gb"] >= 8.0


class TestPlatformOptimizations:
    """Tests for platform-specific performance optimizations."""

    def test_get_platform_specific_optimizations(self):
        """Test getting platform-specific performance optimizations."""
        result = platform_utils.get_platform_specific_optimizations()
        assert isinstance(result, dict)
        assert "use_multiprocessing" in result
        assert "num_workers" in result
        assert "batch_size" in result

    @patch("services_python.platform_utils.is_windows")
    def test_get_platform_specific_optimizations_windows(self, mock_is_windows):
        """Test getting Windows-specific performance optimizations."""
        mock_is_windows.return_value = True
        result = platform_utils.get_platform_specific_optimizations()
        assert result["use_multiprocessing"] is False
        assert result["num_workers"] == 1
        assert result["batch_size"] >= 64

    @patch("services_python.platform_utils.is_linux")
    def test_get_platform_specific_optimizations_linux(self, mock_is_linux):
        """Test getting Linux-specific performance optimizations."""
        mock_is_linux.return_value = True
        with (
            patch("services_python.platform_utils.is_windows", return_value=False),
            patch("services_python.platform_utils.is_macos", return_value=False),
        ):
            result = platform_utils.get_platform_specific_optimizations()
            assert result["use_multiprocessing"] is True
            assert result["num_workers"] >= 8
            assert result["batch_size"] >= 128


class TestGPUBackendPriority:
    """Tests for GPU backend priority based on platform."""

    def test_get_gpu_backend_priority(self):
        """Test getting GPU backend priority."""
        result = platform_utils.get_gpu_backend_priority()
        assert isinstance(result, list)
        assert len(result) > 0
        assert "cpu" in result

    @patch("services_python.platform_utils.is_windows")
    def test_get_gpu_backend_priority_windows(self, mock_is_windows):
        """Test getting GPU backend priority on Windows."""
        mock_is_windows.return_value = True
        result = platform_utils.get_gpu_backend_priority()
        assert result[0] == "cuda"
        assert "directml" in result
        assert "cpu" in result

    @patch("services_python.platform_utils.is_macos")
    def test_get_gpu_backend_priority_macos(self, mock_is_macos):
        """Test getting GPU backend priority on macOS."""
        mock_is_macos.return_value = True
        with (
            patch("services_python.platform_utils.is_windows", return_value=False),
            patch("services_python.platform_utils.is_linux", return_value=False),
        ):
            result = platform_utils.get_gpu_backend_priority()
            assert result[0] == "mps"
            assert "cpu" in result

    @patch("services_python.platform_utils.is_linux")
    def test_get_gpu_backend_priority_linux(self, mock_is_linux):
        """Test getting GPU backend priority on Linux."""
        mock_is_linux.return_value = True
        with (
            patch("services_python.platform_utils.is_windows", return_value=False),
            patch("services_python.platform_utils.is_macos", return_value=False),
        ):
            result = platform_utils.get_gpu_backend_priority()
            assert result[0] == "cuda"
            assert "rocm" in result
            assert "cpu" in result


class TestPlatformCompatibility:
    """Tests for platform compatibility checking."""

    def test_check_platform_compatibility(self):
        """Test platform compatibility checking."""
        result = platform_utils.check_platform_compatibility()
        assert isinstance(result, dict)
        assert "platform" in result
        assert "architecture" in result
        assert "python_version" in result
        assert "compatibility" in result
        assert "warnings" in result
        assert "recommendations" in result

    def test_setup_platform_environment(self):
        """Test platform environment setup."""
        # This should not raise any exceptions
        platform_utils.setup_platform_environment()

        # Check that some environment variables might be set
        # (We can't guarantee which ones, but the function should complete successfully)
        assert True


class TestSafeImport:
    """Tests for safe import functionality."""

    def test_safe_import_success(self):
        """Test successful safe import."""
        result = platform_utils.safe_import("os")
        assert result is not None

    @patch("services_python.platform_utils.is_windows")
    def test_safe_import_windows_specific(self, mock_is_windows):
        """Test safe import of Windows-specific module."""
        mock_is_windows.return_value = False

        # WMI should return None on non-Windows
        result = platform_utils.safe_import("wmi")
        assert result is None

    @patch("services_python.platform_utils.is_windows")
    def test_safe_import_fallback(self, mock_is_windows):
        """Test safe import with fallback."""
        mock_is_windows.return_value = True

        fallback = Mock()
        result = platform_utils.safe_import("nonexistent_module", fallback)
        assert result is fallback

    @patch("services_python.platform_utils.is_windows")
    def test_safe_import_psutil_fallback(self, mock_is_windows):
        """Test safe import of psutil with proper error handling."""
        mock_is_windows.return_value = True

        # This should work on all platforms
        result = platform_utils.safe_import("psutil")
        assert result is not None


class TestPlatformIntegration:
    """Tests for platform integration with other components."""

    def test_platform_constants(self):
        """Test that platform constants are properly set."""
        assert hasattr(platform_utils, "CURRENT_PLATFORM")
        assert hasattr(platform_utils, "CURRENT_ARCH")
        assert hasattr(platform_utils, "PLATFORM_INFO")

        assert isinstance(platform_utils.CURRENT_PLATFORM, platform_utils.Platform)
        assert isinstance(platform_utils.CURRENT_ARCH, platform_utils.Architecture)
        assert isinstance(platform_utils.PLATFORM_INFO, dict)

    def test_platform_info_structure(self):
        """Test platform info structure."""
        info = platform_utils.PLATFORM_INFO
        required_keys = [
            "platform",
            "architecture",
            "python_version",
            "compatibility",
            "warnings",
            "recommendations",
        ]

        for key in required_keys:
            assert key in info

        assert info["compatibility"] in ["excellent", "good", "fair", "poor"]
        assert isinstance(info["warnings"], list)
        assert isinstance(info["recommendations"], list)

    @patch("services_python.platform_utils.Path.exists", return_value=True)
    def test_temp_directory_creation(self, mock_exists):
        """Test temporary directory creation with platform utilities."""
        temp_dir = platform_utils.get_temp_directory()
        assert isinstance(temp_dir, Path)


if __name__ == "__main__":
    pytest.main([__file__])
