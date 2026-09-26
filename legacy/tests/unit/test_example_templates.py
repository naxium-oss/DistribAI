"""Smoke tests for contributor example templates (config loaders)."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example_module(filename: str):
    path = Path(__file__).resolve().parents[2] / "examples" / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def test_train_template_load_config_missing_returns_empty():
    module, path = _load_example_module("train_template.py")
    original = path.parent / "config.json"
    backup = path.parent / "config.json.bak"
    try:
        if original.exists():
            original.rename(backup)
        assert module.load_config() == {}
    finally:
        if backup.exists():
            backup.rename(original)


def test_inference_template_load_config_missing_returns_empty():
    module, path = _load_example_module("inference_template.py")
    original = path.parent / "config.json"
    backup = path.parent / "config.json.bak"
    try:
        if original.exists():
            original.rename(backup)
        assert module.load_config() == {}
    finally:
        if backup.exists():
            backup.rename(original)
