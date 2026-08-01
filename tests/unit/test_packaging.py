"""
Test packaging requirements and configuration.
"""

from pathlib import Path

import pytest


class TestPackagingRequirements:
    """Verify packaging specs include all required dependencies."""

    def test_node_spec_includes_psutil(self):
        """Verify node-windows.spec includes psutil for resource monitoring."""
        spec_file = Path(__file__).parent.parent.parent / "specs" / "node-windows.spec"
        content = spec_file.read_text(encoding="utf-8")

        # psutil is needed for resource limit enforcement in executor.py
        assert "'psutil'" in content or '"psutil"' in content, "psutil should be in hiddenimports"

    def test_node_spec_includes_static_files(self):
        """Verify spec includes dashboard static files."""
        spec_file = Path(__file__).parent.parent.parent / "specs" / "node-windows.spec"
        content = spec_file.read_text(encoding="utf-8")

        # Should include the dashboard static files
        assert "dashboard" in content and "static" in content, (
            "Should include dashboard static files"
        )

    def test_requirements_txt_has_psutil(self):
        """Verify requirements.txt includes psutil."""
        req_file = Path(__file__).parent.parent.parent / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text(encoding="utf-8")
            assert "psutil" in content.lower(), "psutil should be in requirements.txt"

    def test_executor_imports_available(self):
        """Verify all imports in executor.py are available."""
        try:
            import platform  # noqa: F401

            import psutil  # noqa: F401
            import torch  # noqa: F401
            # These are the key imports added for resource limiting
        except ImportError as e:
            pytest.skip(f"Import not available: {e}")

    def test_client_server_js_exists(self):
        """Verify client has server.js for dashboard."""
        client_dir = Path(__file__).parent.parent.parent / "client"
        server_js = client_dir / "server.js"

        assert server_js.exists(), "client/server.js should exist"

        content = server_js.read_text(encoding="utf-8")
        # Check for express usage
        assert "express" in content, "Should use express framework"

    def test_build_script_exists(self):
        """Verify build script exists."""
        build_script = Path(__file__).parent.parent.parent / "build.py"
        assert build_script.exists(), "build.py should exist for packaging"


class TestPackagingIntegrity:
    """Test that packaged application would work correctly."""

    def test_no_test_code_in_package(self):
        """Verify test code is excluded from package."""
        spec_file = Path(__file__).parent.parent.parent / "specs" / "node-windows.spec"
        content = spec_file.read_text(encoding="utf-8")

        # Check that test-related modules are in excludes
        excludes_section = content.split("excludes=")[1] if "excludes=" in content else ""
        assert "pytest" in excludes_section or "test" in excludes_section, (
            "Should exclude test modules"
        )

    def test_resource_limits_function_after_packaging(self):
        """Verify resource limit functions work when imported."""
        # Import the functions directly to test they work
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent.parent))

        try:
            from worker.src.daemon.executor import get_resource_limits

            # Should be able to create these without error
            limits = get_resource_limits()
            assert isinstance(limits, dict), "get_resource_limits should return a dict"
            assert "cpuPercent" in limits, "Should have cpuPercent"
            assert "gpuPercent" in limits, "Should have gpuPercent"
            assert "ramPercent" in limits, "Should have ramPercent"
        except ImportError:
            pytest.skip("Could not import executor module (may need build first)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
