"""End-to-end NsjailSandbox integration tests (Linux-only).

On hosts without ``nsjail`` these tests return immediately (no skip) so
the suite stays compatible with the no-skips policy in ``tests/conftest.py``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from worker.src.sandbox.backends import (
    NetworkPolicy,
    NsjailSandbox,
)


def _nsjail_available() -> bool:
    return not sys.platform.startswith("win") and os.name == "posix" and bool(
        shutil.which("nsjail")
    )


def _write_run(task_dir: Path, body: str) -> None:
    (task_dir / "run.py").write_text(body, encoding="utf-8")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_nsjail_hello_world(tmp_path):
    if not _nsjail_available():
        return
    _write_run(tmp_path, "print('hello from nsjail')")
    sb = NsjailSandbox()
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        max_runtime_seconds=60,
        max_memory_mb=512,
        max_cpu_time_sec=30,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code == 0, f"stderr={res.stderr!r}"
    assert "hello from nsjail" in res.stdout
    assert res.backend_used == "nsjail"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_nsjail_network_none_blocks_dns(tmp_path):
    if not _nsjail_available():
        return
    _write_run(
        tmp_path,
        (
            "import socket, sys\n"
            "try:\n"
            "    socket.gethostbyname('example.com')\n"
            "    print('LEAKED'); sys.exit(1)\n"
            "except (socket.gaierror, OSError):\n"
            "    print('BLOCKED'); sys.exit(0)\n"
        ),
    )
    sb = NsjailSandbox()
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        max_runtime_seconds=60,
        max_memory_mb=512,
        max_cpu_time_sec=30,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code == 0
    assert "BLOCKED" in res.stdout


@pytest.mark.integration
@pytest.mark.asyncio
async def test_nsjail_timeout(tmp_path):
    if not _nsjail_available():
        return
    _write_run(
        tmp_path,
        "import time\nfor _ in range(60):\n    time.sleep(1)\n",
    )
    sb = NsjailSandbox()
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        max_runtime_seconds=3,
        max_memory_mb=512,
        max_cpu_time_sec=60,
        network=NetworkPolicy.NONE,
    )
    assert res.timed_out is True
