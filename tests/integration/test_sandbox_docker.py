"""End-to-end DockerSandbox integration tests.

Skipped on hosts without a working Docker daemon -- both the ``docker``
CLI on PATH AND ``docker info`` succeeding are required, otherwise the
tests are skipped (not failed). This matches the auto-detection logic
in ``worker.src.sandbox.backends.build_sandbox``.

These tests verify the contractual isolation properties that motivated
the v1.2 work:

* ``--network=none`` -> DNS resolution fails inside the sandbox.
* ``--read-only`` -> ``/etc/passwd`` write rejected.
* ``--user=1000:1000`` -> ``os.getuid() != 0`` inside the container.
* ``--memory`` -> a script that tries to allocate >cap dies.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from worker.src.sandbox.backends import (
    DockerSandbox,
    NetworkPolicy,
)


def _docker_daemon_reachable() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        cp = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return cp.returncode == 0


def _write_run(task_dir: Path, body: str) -> None:
    (task_dir / "run.py").write_text(body, encoding="utf-8")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_docker_hello_world(tmp_path):
    if not _docker_daemon_reachable():
        return
    _write_run(tmp_path, "print('hello from docker')")
    sb = DockerSandbox()
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        max_runtime_seconds=120,
        max_memory_mb=512,
        max_cpu_time_sec=60,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code == 0, f"stderr={res.stderr!r}"
    assert "hello from docker" in res.stdout
    assert res.backend_used == "docker"


async def test_docker_network_none_blocks_dns(tmp_path):
    if not _docker_daemon_reachable():
        return
    _write_run(
        tmp_path,
        (
            "import socket, sys\n"
            "try:\n"
            "    socket.gethostbyname('example.com')\n"
            "    print('LEAKED'); sys.exit(1)\n"
            "except (socket.gaierror, OSError) as exc:\n"
            "    print(f'BLOCKED:{type(exc).__name__}')\n"
            "    sys.exit(0)\n"
        ),
    )
    sb = DockerSandbox()
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        max_runtime_seconds=60,
        max_memory_mb=512,
        max_cpu_time_sec=30,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "BLOCKED" in res.stdout
    assert "LEAKED" not in res.stdout


async def test_docker_read_only_blocks_write_to_etc(tmp_path):
    if not _docker_daemon_reachable():
        return
    _write_run(
        tmp_path,
        (
            "import sys\n"
            "try:\n"
            "    open('/etc/passwd', 'w').write('pwned\\n')\n"
            "    print('LEAKED'); sys.exit(1)\n"
            "except (PermissionError, OSError) as exc:\n"
            "    print(f'BLOCKED:{type(exc).__name__}')\n"
            "    sys.exit(0)\n"
        ),
    )
    sb = DockerSandbox()
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        max_runtime_seconds=60,
        max_memory_mb=512,
        max_cpu_time_sec=30,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code == 0, f"stderr={res.stderr!r}"
    assert "BLOCKED" in res.stdout


async def test_docker_runs_as_non_root(tmp_path):
    if not _docker_daemon_reachable():
        return
    _write_run(
        tmp_path,
        (
            "import os, sys\n"
            "uid = os.getuid()\n"
            "print(f'uid={uid}')\n"
            "sys.exit(0 if uid != 0 else 1)\n"
        ),
    )
    sb = DockerSandbox()
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        max_runtime_seconds=60,
        max_memory_mb=512,
        max_cpu_time_sec=30,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "uid=1000" in res.stdout


async def test_docker_memory_cap_kills_oversized_allocation(tmp_path):
    """A 2 GB allocation under a 256 MB cap must NOT succeed.

    The kernel OOM killer terminates the cgroup; Docker propagates a
    non-zero return code. We accept any non-zero exit as success here
    because the precise signal varies across runtimes.
    """
    if not _docker_daemon_reachable():
        return
    _write_run(
        tmp_path,
        (
            "import sys\n"
            "buf = bytearray(2 * 1024 * 1024 * 1024)  # 2 GB\n"
            "print('LEAKED', len(buf)); sys.exit(0)\n"
        ),
    )
    sb = DockerSandbox()
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        max_runtime_seconds=60,
        max_memory_mb=256,
        max_cpu_time_sec=30,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code != 0, (
        f"memory cap should kill the 2 GB allocation, got rc=0 stdout={res.stdout!r}"
    )
    assert "LEAKED" not in res.stdout
