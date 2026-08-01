"""Integration: ScriptRunner converts run.ipynb into executable run.py."""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from worker.src.daemon.script_runner import ScriptRunner


def _ipynb_package() -> bytes:
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    'with open("results.json", "w", encoding="utf-8") as f:\n',
                    '    json.dump({"ok": True, "via": "ipynb"}, f)\n',
                ],
            }
        ],
    }
    raw = json.dumps(notebook).encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="run.ipynb")
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_script_runner_executes_ipynb_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_SANDBOX_BACKEND", "subprocess")
    runner = ScriptRunner(work_dir=tmp_path / "jobs-ipynb")
    result = await runner.execute_task("nb-task-01", _ipynb_package(), {}, {})

    assert result["status"] == "completed"
    assert result.get("results", {}).get("via") == "ipynb"
    assert (tmp_path / "jobs-ipynb" / "nb-task-01" / "run.py").is_file()
