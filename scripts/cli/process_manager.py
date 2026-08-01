"""Background process control for the orchestrator and worker daemon.

Before this module existed, the only documented way to run either process
was a blocking foreground command (see AGENTS.md's "How to run"); there was
no CLI-native way to start one in the background, track it, and stop it
later. This gives ``distribai orchestrator/node {start,stop,status,logs}``
a real (non-mocked) subprocess lifecycle: a PID file plus a rotated-free log
file per named process under ``~/.distribai/run/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path.home() / ".distribai" / "run"


def _pid_file(name: str) -> Path:
    return RUN_DIR / f"{name}.pid"


def _log_file(name: str) -> Path:
    return RUN_DIR / f"{name}.log"


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        if os.name == "nt":
            return True  # best-effort; psutil is a declared dependency in practice
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


class ManagedProcess:
    """Start/stop/status/logs for one named background process."""

    def __init__(self, name: str) -> None:
        self.name = name
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict[str, Any]:
        pid_file = _pid_file(self.name)
        if not pid_file.exists():
            return {"running": False, "pid": None}
        try:
            record = json.loads(pid_file.read_text(encoding="utf-8"))
            pid = int(record.get("pid"))
        except (json.JSONDecodeError, ValueError, OSError, TypeError):
            return {"running": False, "pid": None}
        alive = _pid_alive(pid)
        if not alive:
            pid_file.unlink(missing_ok=True)
        return {
            "running": alive,
            "pid": pid if alive else None,
            "started_at": record.get("started_at"),
            "command": record.get("command"),
            "log_file": str(_log_file(self.name)),
        }

    def start(self, argv: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
        existing = self.status()
        if existing["running"]:
            return {"ok": False, "error": f"{self.name} already running (pid {existing['pid']})"}

        log_path = _log_file(self.name)
        log_handle = open(log_path, "ab", buffering=0)  # noqa: SIM115 - lives for process lifetime
        full_env = {**os.environ, **(env or {})}

        popen_kwargs: dict[str, Any] = {
            "cwd": str(REPO_ROOT),
            "env": full_env,
            "stdout": log_handle,
            "stderr": log_handle,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            )
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen([sys.executable, *argv], **popen_kwargs)
        _pid_file(self.name).write_text(
            json.dumps(
                {
                    "pid": proc.pid,
                    "started_at": int(time.time()),
                    "command": argv,
                }
            ),
            encoding="utf-8",
        )
        return {"ok": True, "pid": proc.pid, "log_file": str(log_path)}

    def stop(self, *, timeout: float = 10.0) -> dict[str, Any]:
        info = self.status()
        if not info["running"]:
            return {"ok": False, "error": f"{self.name} is not running"}
        pid = info["pid"]
        try:
            import psutil

            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except psutil.TimeoutExpired:
                proc.kill()
        except ImportError:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, capture_output=True)
            else:
                os.kill(pid, 15)
        except Exception as exc:  # noqa: BLE001 - report and continue cleanup
            _pid_file(self.name).unlink(missing_ok=True)
            return {"ok": False, "error": str(exc)}
        _pid_file(self.name).unlink(missing_ok=True)
        return {"ok": True, "pid": pid}

    def tail_logs(self, lines: int = 60) -> list[str]:
        log_path = _log_file(self.name)
        if not log_path.exists():
            return []
        with open(log_path, "rb") as fh:
            content = fh.read().decode("utf-8", errors="replace")
        return content.splitlines()[-lines:]


def orchestrator_argv(grpc_port: int | None, admin_port: int | None) -> list[str]:
    return ["-m", "services_python.orchestrator_grpc"]


def orchestrator_env(grpc_port: int | None, admin_port: int | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if grpc_port:
        env["GRPC_PORT"] = str(grpc_port)
    if admin_port:
        env["ADMIN_PORT"] = str(admin_port)
    return env


def worker_argv(
    orchestrator_url: str | None, node_id: str | None, worker_index: int | None
) -> list[str]:
    argv = ["-m", "worker.src.daemon.run"]
    if orchestrator_url:
        argv += ["--orchestrator", orchestrator_url]
    if node_id:
        argv += ["--node-id", node_id]
    if worker_index is not None:
        argv += ["--worker-index", str(worker_index)]
    return argv
