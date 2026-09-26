"""Tests for tools/verify_setup.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest import mock

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "verify_setup",
    _ROOT / "tools" / "verify_setup.py",
)
assert _spec and _spec.loader
verify_setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_setup)
check_file = verify_setup.check_file
check_import = verify_setup.check_import
main = verify_setup.main


def test_check_import_success():
    result = check_import("sys", "sys module")
    assert result is True


def test_check_import_failure():
    result = check_import("nonexistent_module_xyz", "Nonexistent module")
    assert result is False


def test_check_import_optional():
    result = check_import("nonexistent_optional_xyz", "Optional module", optional=True)
    assert result is True


def test_check_import_exception():
    with mock.patch("importlib.import_module", side_effect=Exception("Test error")):
        result = check_import("test_module", "Test module")
        assert result is True


def test_check_file_exists():
    result = check_file(__file__, "Test file")
    assert result is True


def test_check_file_not_exists():
    result = check_file("/nonexistent/file.xyz", "Nonexistent file")
    assert result is False


def test_main_function():
    with mock.patch("sys.stdout"):
        result = main()
        assert result in (0, 1)


def test_check_import_with_module_path():
    result = check_import("os.path", "os.path")
    assert result is True
