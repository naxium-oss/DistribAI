"""Unit tests for notebook → run.py mount helpers."""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from services_python.preflight import validate_script_tarball
from worker.src.sandbox.notebook_mount import (
    NotebookMountError,
    ensure_run_py,
    ipynb_to_python,
)


def _sample_notebook() -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Demo\n"]},
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import json\n",
                    'with open("results.json", "w", encoding="utf-8") as f:\n',
                    '    json.dump({"from_notebook": True}, f)\n',
                ],
                "outputs": [],
            },
        ],
    }


def test_ipynb_to_python_extracts_code_cells_only():
    text = ipynb_to_python(_sample_notebook())
    assert "from_notebook" in text
    assert "# Demo" not in text
    assert "import json" in text


def test_ipynb_to_python_rejects_empty_code():
    empty = {
        "nbformat": 4,
        "cells": [{"cell_type": "markdown", "source": "hi", "metadata": {}}],
    }
    with pytest.raises(NotebookMountError, match="no code cells"):
        ipynb_to_python(empty)


def test_ensure_run_py_converts_notebook(tmp_path):
    nb_path = tmp_path / "run.ipynb"
    nb_path.write_text(json.dumps(_sample_notebook()), encoding="utf-8")
    run_py = ensure_run_py(tmp_path)
    assert run_py.is_file()
    assert "from_notebook" in run_py.read_text(encoding="utf-8")


def test_preflight_accepts_run_ipynb_only():
    body = json.dumps(_sample_notebook()).encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="run.ipynb")
        info.size = len(body)
        tar.addfile(info, io.BytesIO(body))
    ok, err, meta = validate_script_tarball(buf.getvalue())
    assert ok is True
    assert err is None
    assert meta["member_count"] == 1
