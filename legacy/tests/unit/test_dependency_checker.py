"""Tests for dependency_checker module."""

# Import the module under test
import services_python.dependency_checker as dep_checker


class TestDependencyChecker:
    """Test cases for dependency checker functionality."""

    def test_module_import(self):
        """Test that the module can be imported."""
        assert hasattr(dep_checker, "SafetyLevel")
        assert hasattr(dep_checker, "DependencyCheck")
        assert hasattr(dep_checker, "MALICIOUS_PACKAGES")
        assert hasattr(dep_checker, "SYSTEM_PACKAGES")
        assert hasattr(dep_checker, "DANGEROUS_PACKAGES")
        assert hasattr(dep_checker, "POPULAR_PACKAGES")
        assert hasattr(dep_checker, "quick_check")
        assert hasattr(dep_checker, "check_requirements_list")

    def test_safety_level_enum(self):
        """Test SafetyLevel enum values."""
        assert dep_checker.SafetyLevel.SAFE.value == "safe"
        assert dep_checker.SafetyLevel.WARNING.value == "warning"
        assert dep_checker.SafetyLevel.BLOCKED.value == "blocked"

    def test_dependency_check_dataclass(self):
        """Test DependencyCheck dataclass."""
        check = dep_checker.DependencyCheck(
            package_name="test-package",
            version_spec=">=1.0.0",
            safety_level=dep_checker.SafetyLevel.SAFE,
            reason="Test package",
            suggestions=["alt-package"],
            requires_admin_approval=False,
        )

        assert check.package_name == "test-package"
        assert check.version_spec == ">=1.0.0"
        assert check.safety_level == dep_checker.SafetyLevel.SAFE
        assert check.reason == "Test package"
        assert check.suggestions == ["alt-package"]
        assert check.requires_admin_approval is False

    def test_check_dependency_malicious_package(self):
        """Test checking a known malicious package."""
        result = dep_checker.quick_check("reqeusts")

        assert result.package_name == "reqeusts"
        assert result.safety_level == dep_checker.SafetyLevel.BLOCKED
        assert "malicious" in result.reason.lower() or "typosquat" in result.reason.lower()
        assert result.requires_admin_approval is False

    def test_check_dependency_system_package(self):
        """Test checking a system package."""
        result = dep_checker.quick_check("os")

        assert result.package_name == "os"
        assert result.safety_level == dep_checker.SafetyLevel.BLOCKED
        assert "standard library" in result.reason.lower() or "system" in result.reason.lower()
        assert result.requires_admin_approval is False

    def test_check_dependency_dangerous_package(self):
        """Test checking a dangerous package."""
        result = dep_checker.quick_check("requests")

        assert result.package_name == "requests"
        assert result.safety_level == dep_checker.SafetyLevel.WARNING
        assert "network" in result.reason.lower()
        assert result.requires_admin_approval is True

    def test_check_dependency_safe_package(self):
        """Test checking a safe package."""
        result = dep_checker.quick_check("numpy")

        assert result.package_name == "numpy"
        assert result.safety_level == dep_checker.SafetyLevel.SAFE
        assert result.requires_admin_approval is False

    def test_check_dependency_typosquat_detection(self):
        """Test typosquatting detection."""
        # Test a typo of a popular package
        result = dep_checker.quick_check("numpyy")

        assert result.package_name == "numpyy"
        assert result.safety_level == dep_checker.SafetyLevel.BLOCKED
        assert "typosquat" in result.reason.lower()
        assert "numpy" in result.suggestions
        assert result.requires_admin_approval is True

    def test_check_dependency_with_version_spec(self):
        """Test checking dependency with version specification."""
        version_spec = ">=1.0.0,<2.0.0"
        result = dep_checker.quick_check(f"numpy{version_spec}")

        assert result.package_name == "numpy"
        assert result.version_spec == version_spec
        assert result.safety_level == dep_checker.SafetyLevel.SAFE

    def test_check_dependencies_list(self):
        """Test checking multiple dependencies."""
        dependencies = [
            "numpy>=1.0.0",
            "reqeusts",  # malicious
            "os",  # system package
        ]

        results = dep_checker.check_requirements_list(dependencies)

        assert "safe" in results
        assert "blocked" in results

        safe_names = [c.package_name for c in results["safe"]]
        blocked_names = [c.package_name for c in results["blocked"]]

        assert "numpy" in safe_names
        assert "reqeusts" in blocked_names
        assert "os" in blocked_names
        assert results["can_proceed"] is False

    def test_package_name_normalization(self):
        """Test package name normalization."""
        # Test case normalization
        result1 = dep_checker.quick_check("NUMPY")
        result2 = dep_checker.quick_check("numpy")

        assert result1.safety_level == result2.safety_level

    def test_version_spec_parsing(self):
        """Test version specification parsing."""
        # Test various version specs
        test_cases = [
            ">=1.0.0",
            ">=1.0.0,<2.0.0",
            "==1.2.3",
            "~=1.2.0",
            ">=1.0.0,!=1.2.0",
        ]

        for version_spec in test_cases:
            result = dep_checker.quick_check(f"numpy{version_spec}")
            assert result.version_spec == version_spec
            assert result.package_name == "numpy"

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty package name
        result = dep_checker.quick_check("")
        assert result.package_name == ""
        assert result.safety_level == dep_checker.SafetyLevel.SAFE  # Empty is probably safe

        # None version spec (already tested in other tests)
        result = dep_checker.quick_check("numpy")
        assert result.version_spec is None

        # Very long package name
        long_name = "a" * 1000
        result = dep_checker.quick_check(long_name)
        assert result.package_name == long_name
        assert result.safety_level == dep_checker.SafetyLevel.SAFE  # Unknown package is safe
