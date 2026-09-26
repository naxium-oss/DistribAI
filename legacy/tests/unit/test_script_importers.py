"""Smoke-import CI and dev scripts so test_coverage sees importers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPTS = [
    "scripts/ci/find_python.py",
    "scripts/ci/check_bandit_gate.py",
    "scripts/ci/check_pip_audit_gate.py",
    "scripts/ci/check_safety_gate.py",
    "scripts/dev/mini_smoke.py",
]


def _load_script(rel_path: str):
    path = Path(__file__).resolve().parents[2] / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_python_get_python_command():
    mod = _load_script("scripts/ci/find_python.py")
    cmd = mod.get_python_command()
    assert isinstance(cmd, list)
    assert cmd


@pytest.mark.parametrize("rel_path", _SCRIPTS)
def test_script_modules_expose_main_or_helpers(rel_path: str):
    mod = _load_script(rel_path)
    assert mod is not None
