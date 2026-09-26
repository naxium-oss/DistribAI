"""Turn a join link into a worker start - the contributor side.

Works the same on a desktop with a local GPU, in Colab, Kaggle, Molab, or any
VPS. The link decides the engine:

* ``native`` (the default): the C++ worker, which registers over HTTP, claims a
  replica, runs the LibTorch trainer as a limited child, and reports a
  GradReport envelope. Build it with ``make grid``.
* ``legacy``: the frozen Python gRPC worker daemon, kept for grids that still
  run the old control plane.

Either way the plan is the same shape: env, argv, notes. ``join(..., execute=True)``
runs it.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .link import GridLink

WORKER_MODULE = "worker.src.daemon.run"
WORKER_BINARY = "build/cpp_port/distribai_worker"


@dataclass
class JoinPlan:
    link: GridLink
    env: dict[str, str]
    argv: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def engine(self) -> str:
        return self.link.engine

    @property
    def orchestrator_url(self) -> str:
        return self.link.orchestrator_url

    def env_lines(self) -> list[str]:
        return [f"{k}={v}" for k, v in self.env.items()]

    def shell(self) -> str:
        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        return f"{exports} {' '.join(shlex.quote(a) for a in self.argv)}".strip()

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "orchestrator_url": self.orchestrator_url,
            "tls": self.link.tls,
            "provider": self.link.provider,
            "env": dict(self.env),
            "argv": list(self.argv),
            "notes": list(self.notes),
        }

    def render(self) -> str:
        lines = [
            f"engine       : {self.engine}",
            f"orchestrator : {self.orchestrator_url}  "
            f"(provider={self.link.provider}, tls={self.link.tls})",
            "environment  :",
        ]
        lines += [f"    {line}" for line in self.env_lines()] or ["    (none)"]
        lines.append("command      :")
        lines.append(f"    {' '.join(shlex.quote(a) for a in self.argv)}")
        if self.notes:
            lines.append("notes        :")
            lines += [f"    - {note}" for note in self.notes]
        return "\n".join(lines)


def plan_join(
    target: str | GridLink,
    *,
    node_id: str | None = None,
    state_dir: str | Path | None = None,
    ephemeral: bool = False,
    invite: str | None = None,
    admin_url: str | None = None,
    python: str | None = None,
    engine: str | None = None,
    worker_bin: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> JoinPlan:
    """Build the worker env and argv for a join link. Has no side effects."""
    link = target if isinstance(target, GridLink) else GridLink.parse(target)
    if engine:
        link.engine = engine
    python = python or sys.executable
    invite = invite or link.invite
    admin = admin_url or link.admin_url

    env: dict[str, str] = {"ORCHESTRATOR_URL": link.orchestrator_url}
    if invite:
        env["DISTRIBAI_INVITE_CODE"] = invite
    if admin:
        env["ADMIN_URL"] = admin
    if ephemeral:
        env["DISTRIBAI_EPHEMERAL"] = "1"
    if state_dir is not None:
        env["STATE_DIR"] = str(state_dir)

    notes: list[str] = []

    if link.is_native:
        binary = str(worker_bin or WORKER_BINARY)
        argv = [binary, "--orchestrator", link.orchestrator_url]
        if invite:
            argv += ["--invite", invite]
        if node_id:
            argv += ["--node-id", node_id]
        if state_dir is not None:
            argv += ["--work-dir", str(state_dir)]
        notes.append(
            "the native worker needs build/cpp_port/distribai_worker (make grid) and the "
            "LibTorch trainer for the job it claims (make torch)."
        )
        notes.append(
            "it registers over HTTP and reports a GradReport envelope per replica; the "
            "coordinator aggregates them."
        )
    else:
        env["GRPC_USE_TLS"] = "true" if link.tls else "false"
        argv = [python, "-m", WORKER_MODULE, "--orchestrator", link.orchestrator_url]
        if node_id:
            argv += ["--node-id", node_id]
        notes.append("legacy gRPC path: the frozen Python stack answers this link.")

    if link.tls:
        if not link.is_native and os.environ.get("GRPC_TLS_CA"):
            notes.append(f"using GRPC_TLS_CA={os.environ['GRPC_TLS_CA']}")
        elif link.provider in {"cloudflare", "ngrok"}:
            notes.append(
                "the tunnel endpoint uses a public certificate, so the system trust store is "
                "enough; set GRPC_TLS_CA only for a private orchestrator CA."
            )
        else:
            notes.append("TLS is on; point at the CA if the orchestrator uses a private one.")
    else:
        notes.append("plaintext on the wire: fine on loopback or a LAN, not on the open internet.")

    if ephemeral:
        notes.append("ephemeral mode: use a scratch work directory (Colab, Kaggle, Molab).")
    if link.provider == "local" and link.host not in {"127.0.0.1", "::1", "localhost"}:
        notes.append(
            f"{link.host} looks like a LAN address; check it is reachable from this machine."
        )
    if extra_env:
        env.update(extra_env)
    return JoinPlan(link=link, env=env, argv=argv, notes=notes)


def join(
    target: str | GridLink,
    *,
    execute: bool = False,
    cwd: str | Path | None = None,
    **plan_kwargs,
) -> JoinPlan:
    """Plan a join, optionally launching the worker with the plan's env."""
    plan = plan_join(target, **plan_kwargs)
    if not execute:
        return plan
    env = {**os.environ, **plan.env}
    subprocess.run(plan.argv, env=env, cwd=str(cwd) if cwd else None, check=False)
    return plan


__all__ = ["JoinPlan", "join", "plan_join"]
