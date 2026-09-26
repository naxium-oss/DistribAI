"""Unit tests for distribai_cli submit bundling."""

from __future__ import annotations

import hashlib

import pytest

from scripts.cli.distribai_cli import JobManager


@pytest.mark.unit
def test_bundle_directory_requires_run_py(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "other.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run.py or run.ipynb"):
        JobManager.bundle_directory(folder)


@pytest.mark.unit
def test_bundle_directory_accepts_run_ipynb(tmp_path):
    folder = tmp_path / "nbproj"
    folder.mkdir()
    notebook = {
        "nbformat": 4,
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": ["print('hi')\n"],
                "outputs": [],
            }
        ],
        "metadata": {},
    }
    import json

    (folder / "run.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    raw, digest = JobManager.bundle_directory(folder)
    assert len(raw) > 0
    assert len(digest) == 64


@pytest.mark.unit
def test_bundle_directory_produces_stable_digest(tmp_path):
    folder = tmp_path / "proj"
    folder.mkdir()
    (folder / "run.py").write_text(
        "import json\njson.dump({'ok': True}, open('results.json', 'w'))\n",
        encoding="utf-8",
    )
    raw, digest = JobManager.bundle_directory(folder)
    assert len(raw) > 0
    assert digest == hashlib.sha256(raw).hexdigest()
