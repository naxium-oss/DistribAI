"""Script runner for executing job tasks on nodes.

Handles unpacking and execution of job scripts from the server.
Tar members are validated (no `../`, symlinks, or device nodes) before extract.
Corrupted archives surface as ``failed`` with an ``error`` string in the result dict.
"""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path


def _unpack_max_retries() -> int:
    raw = os.getenv("SCRIPT_UNPACK_MAX_RETRIES", "2").strip()
    try:
        return max(1, min(5, int(raw)))
    except ValueError:
        return 2


def _egress_denied(hyperparams: dict, env_vars: dict) -> bool:
    flag = os.getenv("DISTRIBAI_DENY_EGRESS", "").strip().lower()
    if flag in ("1", "true", "yes"):
        return True
    return bool(hyperparams.get("deny_egress"))


def _network_policy(hyperparams: dict, env_vars: dict):
    from worker.src.sandbox.backends.base import NetworkPolicy

    raw = (
        hyperparams.get("network_policy")
        or env_vars.get("DISTRIBAI_NETWORK_POLICY")
        or os.getenv("DISTRIBAI_NETWORK_POLICY", "")
    )
    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in ("open", "restricted", "none"):
            return NetworkPolicy(key)
    if _egress_denied(hyperparams, env_vars):
        return NetworkPolicy.NONE
    return NetworkPolicy.OPEN


def _resource_limits(hyperparams: dict) -> tuple[int, int, int]:
    try:
        max_runtime = int(hyperparams.get("max_runtime_seconds", 3600))
    except (TypeError, ValueError):
        max_runtime = 3600
    max_runtime = max(1, min(max_runtime, 86400))
    try:
        max_memory_mb = int(hyperparams.get("max_memory_mb", 8192))
    except (TypeError, ValueError):
        max_memory_mb = 8192
    max_memory_mb = max(256, min(max_memory_mb, 65536))
    try:
        max_cpu_time_sec = int(hyperparams.get("max_cpu_time_sec", max_runtime))
    except (TypeError, ValueError):
        max_cpu_time_sec = max_runtime
    max_cpu_time_sec = max(1, min(max_cpu_time_sec, max_runtime))
    return max_runtime, max_memory_mb, max_cpu_time_sec


class ScriptRunner:
    """Executes job scripts on the node."""

    def __init__(self, work_dir: Path | None = None):
        self.work_dir = work_dir or Path.home() / ".distribai" / "jobs"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.active_processes: dict[str, subprocess.Popen] = {}

    async def execute_task(
        self, task_id: str, script_package: bytes, env_vars: dict, hyperparams: dict
    ) -> dict:
        """Execute a task script package."""
        task_dir = self.work_dir / task_id
        task_dir.mkdir(exist_ok=True)

        try:
            package_digest = hashlib.sha256(script_package).hexdigest()
            expected_digest = hyperparams.get("package_sha256") or env_vars.get(
                "DISTRIBAI_PACKAGE_SHA256"
            )
            if expected_digest and expected_digest != package_digest:
                return {
                    "status": "failed",
                    "error": "script package checksum mismatch",
                }

            package_path = task_dir / "package.tar.gz"
            with open(package_path, "wb") as f:
                f.write(script_package)
            if hashlib.sha256(package_path.read_bytes()).hexdigest() != package_digest:
                return {
                    "status": "failed",
                    "error": "script package checksum failed after write",
                }

            def _safe_tar_member(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
                member_path = Path(str(member.name).replace("\\", "/"))
                if not member_path.parts or ".." in member_path.parts or member_path.is_absolute():
                    return None
                target = Path(dest) / member_path
                try:
                    target.resolve().relative_to(Path(dest).resolve())
                except ValueError:
                    return None
                if member.issym() or member.islnk() or member.isdev():
                    return None
                return member

            unpack_error: str | None = None
            for attempt in range(_unpack_max_retries()):
                try:
                    with tarfile.open(package_path, "r:gz") as tar:
                        dest_str = str(task_dir)
                        safe_members: list[tarfile.TarInfo] = []
                        for member in tar.getmembers():
                            safe_member = _safe_tar_member(member, dest_str)
                            if safe_member is None:
                                return {
                                    "status": "failed",
                                    "error": f"Invalid tar member rejected: {member.name}",
                                }
                            safe_members.append(safe_member)
                        for safe_member in safe_members:
                            tar.extract(safe_member, task_dir, filter="data")
                    unpack_error = None
                    break
                except (OSError, tarfile.ReadError) as exc:
                    unpack_error = str(exc)
                    if attempt + 1 < _unpack_max_retries():
                        time.sleep(0.05 * (attempt + 1))
            if unpack_error is not None:
                return {
                    "status": "failed",
                    "error": f"Failed to unpack script package after retries: {unpack_error}",
                }

            # Load config
            config_path = task_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = {}

            # Set environment variables
            env = os.environ.copy()
            env.update(env_vars)
            env.update(
                {
                    "DISTRIBAI_TASK_ID": task_id,
                    "DISTRIBAI_JOB_ID": config.get("job_id") or env.get("DISTRIBAI_JOB_ID", ""),
                    "DISTRIBAI_JOB_TYPE": config.get("job_type", ""),
                }
            )

            # Write hyperparams to file for script access
            hyperparams_path = task_dir / "hyperparams.json"
            with open(hyperparams_path, "w") as f:
                json.dump(hyperparams, f)

            # Materialize run.py from run.ipynb when the bundle ships a notebook entrypoint
            from worker.src.sandbox.notebook_mount import NotebookMountError, ensure_run_py

            try:
                script_path = ensure_run_py(task_dir)
            except NotebookMountError as exc:
                return {
                    "status": "failed",
                    "error": str(exc) if str(exc) else "No run.py found in package",
                }
            if not script_path.exists():
                return {
                    "status": "failed",
                    "error": "No run.py found in package",
                }

            # Install requirements if present (isolated to task directory)
            req_path = task_dir / "requirements.txt"
            if req_path.exists() and _egress_denied(hyperparams, env_vars):
                return {
                    "status": "failed",
                    "error": "Network egress denied by policy; cannot install requirements.txt",
                }
            if req_path.exists():
                # Create isolated site-packages directory for this task
                site_packages_dir = task_dir / ".site-packages"
                site_packages_dir.mkdir(exist_ok=True)

                install_result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(req_path),
                        "--target",
                        str(site_packages_dir),
                        "--no-user",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 min timeout for installs
                )
                if install_result.returncode != 0:
                    return {
                        "status": "failed",
                        "error": f"Requirements install failed: {install_result.stderr}",
                    }

                # Add isolated site-packages to PYTHONPATH for this task only
                env["PYTHONPATH"] = str(site_packages_dir) + os.pathsep + env.get("PYTHONPATH", "")

            max_runtime, max_memory_mb, max_cpu_time_sec = _resource_limits(hyperparams)
            network = _network_policy(hyperparams, env_vars)

            from worker.src.sandbox.backends import build_sandbox

            backend_override = (
                hyperparams.get("sandbox_backend")
                or env_vars.get("DISTRIBAI_SANDBOX_BACKEND")
                or os.getenv("DISTRIBAI_SANDBOX_BACKEND")
            )
            sandbox = build_sandbox(backend=backend_override)

            def _register_process(proc: subprocess.Popen) -> None:
                self.active_processes[task_id] = proc

            try:
                sb_result = await sandbox.run_script(
                    task_dir=task_dir,
                    env=env,
                    max_runtime_seconds=max_runtime,
                    max_memory_mb=max_memory_mb,
                    max_cpu_time_sec=max_cpu_time_sec,
                    network=network,
                    on_process_started=_register_process,
                )
            finally:
                self.active_processes.pop(task_id, None)

            if sb_result.timed_out:
                return {
                    "status": "failed",
                    "error": f"Script execution exceeded {max_runtime} seconds timeout",
                    "backend_used": sb_result.backend_used,
                }

            if sb_result.return_code == -1 and sb_result.stderr.startswith("run.py missing"):
                return {
                    "status": "failed",
                    "error": sb_result.stderr,
                    "backend_used": sb_result.backend_used,
                }

            result = {
                "status": "completed" if sb_result.return_code == 0 else "failed",
                "return_code": sb_result.return_code,
                "stdout": sb_result.stdout,
                "stderr": sb_result.stderr,
                "backend_used": sb_result.backend_used,
            }

            # Look for output files
            checkpoint_path = task_dir / "checkpoint.pt"
            if checkpoint_path.exists():
                result["has_checkpoint"] = True
                result["checkpoint_path"] = str(checkpoint_path)

            results_path = task_dir / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    result["results"] = json.load(f)

            metrics_path = task_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    result["metrics"] = json.load(f)

            return result

        except subprocess.TimeoutExpired:
            if task_id in self.active_processes:
                process = self.active_processes.pop(task_id)
                process.terminate()
            return {
                "status": "failed",
                "error": "Task timed out",
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
            }

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id in self.active_processes:
            process = self.active_processes.pop(task_id)
            process.terminate()
            return True
        return False

    def cleanup_task(self, task_id: str):
        """Clean up task directory."""
        task_dir = self.work_dir / task_id
        if task_dir.exists():
            import shutil

            shutil.rmtree(task_dir)
