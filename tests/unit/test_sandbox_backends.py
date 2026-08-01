"""Unit tests for the v1.2 sandbox backend abstraction.

Three concerns:

1. ``detect_backend`` returns the right name under the right host
   conditions (env override, docker on PATH, nsjail on PATH, neither).
2. ``build_sandbox`` falls back gracefully when the requested backend
   is unavailable.
3. ``SubprocessSandbox`` actually executes scripts and enforces the
   v1.1 invariants (env whitelist, timeout, working directory).

Docker / nsjail integration tests live in
``tests/integration/test_sandbox_*`` so this file stays fast and
deterministic on every host.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from worker.src.sandbox.backends import (
    DockerSandbox,
    NetworkPolicy,
    NsjailSandbox,
    SandboxResult,
    SubprocessSandbox,
    build_sandbox,
    detect_backend,
)

# ---------------------------------------------------------------------------
# detect_backend
# ---------------------------------------------------------------------------


def test_detect_backend_subprocess_when_neither_available(monkeypatch):
    """No docker, no nsjail -> 'subprocess'."""
    monkeypatch.delenv("DISTRIBAI_SANDBOX_BACKEND", raising=False)
    with patch("worker.src.sandbox.backends.shutil.which", return_value=None):
        assert detect_backend() == "subprocess"


def test_detect_backend_docker_when_docker_available(monkeypatch):
    """Docker on PATH -> 'docker' even if nsjail is also there."""
    monkeypatch.delenv("DISTRIBAI_SANDBOX_BACKEND", raising=False)

    def fake_which(name):
        return f"/usr/bin/{name}" if name in {"docker", "nsjail"} else None

    with patch("worker.src.sandbox.backends.shutil.which", side_effect=fake_which):
        assert detect_backend() == "docker"


def test_detect_backend_nsjail_when_only_nsjail_available(monkeypatch):
    """nsjail on PATH, no docker, POSIX -> 'nsjail'."""
    monkeypatch.delenv("DISTRIBAI_SANDBOX_BACKEND", raising=False)

    def fake_which(name):
        return "/usr/bin/nsjail" if name == "nsjail" else None

    with (
        patch("worker.src.sandbox.backends.shutil.which", side_effect=fake_which),
        patch("worker.src.sandbox.backends.os.name", "posix"),
    ):
        assert detect_backend() == "nsjail"


def test_detect_backend_env_override_subprocess(monkeypatch):
    """DISTRIBAI_SANDBOX_BACKEND=subprocess wins over Docker on PATH."""
    monkeypatch.setenv("DISTRIBAI_SANDBOX_BACKEND", "subprocess")
    with patch("worker.src.sandbox.backends.shutil.which", return_value="/usr/bin/docker"):
        assert detect_backend() == "subprocess"


def test_detect_backend_env_override_docker(monkeypatch):
    """DISTRIBAI_SANDBOX_BACKEND=docker wins even when docker is not on PATH."""
    monkeypatch.setenv("DISTRIBAI_SANDBOX_BACKEND", "docker")
    with patch("worker.src.sandbox.backends.shutil.which", return_value=None):
        assert detect_backend() == "docker"


def test_detect_backend_env_override_invalid_falls_through(monkeypatch):
    """Garbage env values are ignored, autodetection wins."""
    monkeypatch.setenv("DISTRIBAI_SANDBOX_BACKEND", "bogus-backend")
    with patch("worker.src.sandbox.backends.shutil.which", return_value=None):
        assert detect_backend() == "subprocess"


# ---------------------------------------------------------------------------
# build_sandbox fallback
# ---------------------------------------------------------------------------


def test_build_sandbox_docker_falls_back_when_daemon_unreachable(monkeypatch):
    """Asking for docker but is_available() is False -> SubprocessSandbox."""
    monkeypatch.delenv("DISTRIBAI_SANDBOX_BACKEND", raising=False)
    with (
        patch.object(DockerSandbox, "is_available", return_value=False),
        patch("worker.src.sandbox.backends.shutil.which", return_value=None),
    ):
        sb = build_sandbox(backend="docker")
        assert isinstance(sb, SubprocessSandbox)


def test_build_sandbox_subprocess_always_works():
    sb = build_sandbox(backend="subprocess")
    assert isinstance(sb, SubprocessSandbox)
    assert sb.name == "subprocess"


# ---------------------------------------------------------------------------
# SubprocessSandbox end-to-end behaviour
# ---------------------------------------------------------------------------


def _write_hello_run_py(task_dir: Path, body: str = "print('hello from sandbox')") -> None:
    (task_dir / "run.py").write_text(body, encoding="utf-8")


async def test_subprocess_runs_hello_world(tmp_path):
    sb = SubprocessSandbox()
    _write_hello_run_py(tmp_path)
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1"},
        max_runtime_seconds=15,
        max_memory_mb=1024,
        max_cpu_time_sec=10,
        network=NetworkPolicy.NONE,
    )
    assert isinstance(res, SandboxResult)
    assert res.return_code == 0, f"stderr={res.stderr!r}"
    assert "hello from sandbox" in res.stdout
    assert res.timed_out is False
    assert res.backend_used == "subprocess"


async def test_subprocess_returns_run_py_missing_error(tmp_path):
    sb = SubprocessSandbox()
    res = await sb.run_script(
        task_dir=tmp_path,
        env={},
        max_runtime_seconds=5,
        max_memory_mb=512,
        max_cpu_time_sec=5,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code == -1
    assert "run.py missing" in res.stderr


async def test_subprocess_propagates_nonzero_exit(tmp_path):
    sb = SubprocessSandbox()
    _write_hello_run_py(
        tmp_path,
        body="import sys; sys.stderr.write('boom\\n'); sys.exit(7)",
    )
    res = await sb.run_script(
        task_dir=tmp_path,
        env={},
        max_runtime_seconds=10,
        max_memory_mb=512,
        max_cpu_time_sec=10,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code == 7
    assert "boom" in res.stderr


async def test_subprocess_enforces_timeout(tmp_path):
    sb = SubprocessSandbox()
    _write_hello_run_py(
        tmp_path,
        body="import time\nfor _ in range(30):\n    time.sleep(1)\n",
    )
    res = await sb.run_script(
        task_dir=tmp_path,
        env={},
        max_runtime_seconds=2,
        max_memory_mb=512,
        max_cpu_time_sec=30,
        network=NetworkPolicy.NONE,
    )
    assert res.timed_out is True


async def test_subprocess_enforces_env_whitelist(tmp_path, monkeypatch):
    """Backend MUST NOT leak the daemon's os.environ.

    The script is launched with exactly the env dict we hand it. We
    plant SECRET_FOR_TEST in the daemon's environment, then verify the
    child cannot see it (the script writes its environment to disk).
    """
    monkeypatch.setenv("SECRET_FOR_TEST", "never-show-up-in-child")

    probe = tmp_path / "env_dump.txt"
    probe_path_literal = str(probe).replace("\\", "\\\\")
    _write_hello_run_py(
        tmp_path,
        body=(
            "import os, json\n"
            f"open(r'{probe_path_literal}', 'w').write("
            "json.dumps(dict(os.environ)))\n"
        ),
    )

    sb = SubprocessSandbox()
    # Crucially: pass a minimal env dict, NOT os.environ.
    res = await sb.run_script(
        task_dir=tmp_path,
        env={"PYTHONUNBUFFERED": "1", "PATH": os.environ.get("PATH", "")},
        max_runtime_seconds=15,
        max_memory_mb=1024,
        max_cpu_time_sec=10,
        network=NetworkPolicy.NONE,
    )
    assert res.return_code == 0, f"stderr={res.stderr!r}"

    import json as _json

    child_env = _json.loads(probe.read_text(encoding="utf-8"))
    assert "SECRET_FOR_TEST" not in child_env, (
        "SubprocessSandbox leaked a daemon-env secret into the child process. "
        "The v1.1 env whitelist is the gate -- it must not be bypassed."
    )


async def test_subprocess_open_network_policy_does_not_warn_loudly(tmp_path, caplog):
    """OPEN policy should NOT emit the 'cannot enforce' warning."""
    sb = SubprocessSandbox()
    _write_hello_run_py(tmp_path)
    import logging as _logging

    caplog.set_level(_logging.WARNING, logger="worker.src.sandbox.backends.subprocess_backend")
    res = await sb.run_script(
        task_dir=tmp_path,
        env={},
        max_runtime_seconds=10,
        max_memory_mb=512,
        max_cpu_time_sec=10,
        network=NetworkPolicy.OPEN,
    )
    assert res.return_code == 0
    assert not any("cannot enforce" in r.message for r in caplog.records)


async def test_subprocess_none_network_policy_warns(tmp_path, caplog):
    sb = SubprocessSandbox()
    _write_hello_run_py(tmp_path)
    import logging as _logging

    caplog.set_level(_logging.WARNING, logger="worker.src.sandbox.backends.subprocess_backend")
    await sb.run_script(
        task_dir=tmp_path,
        env={},
        max_runtime_seconds=10,
        max_memory_mb=512,
        max_cpu_time_sec=10,
        network=NetworkPolicy.NONE,
    )
    assert any("cannot enforce NetworkPolicy.NONE" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Smoke: Docker/Nsjail constructors do not raise on import
# ---------------------------------------------------------------------------


def test_docker_sandbox_can_be_instantiated_anywhere():
    sb = DockerSandbox()
    assert sb.name == "docker"
    assert sb.image  # default or env-overridden


def test_nsjail_sandbox_can_be_instantiated_anywhere():
    sb = NsjailSandbox()
    assert sb.name == "nsjail"


def test_nsjail_is_unavailable_on_non_posix(monkeypatch):
    """nsjail backend MUST report unavailable on Windows."""
    if os.name == "posix":
        pytest.skip("POSIX host -- this test exercises the Windows path only")
    sb = NsjailSandbox()
    assert sb.is_available() is False
