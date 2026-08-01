#!/usr/bin/env python3
"""
DistribAI Command Line Interface

Management tool for DistribAI nodes, orchestrators, and jobs.
"""

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import webbrowser
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_on_path() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# This file is also invoked as a bare script (`python scripts/cli/distribai_cli.py`),
# so the repo root must land on sys.path before these sibling-package imports resolve.
_ensure_repo_on_path()

from scripts.cli.api_client import AdminAPIClient  # noqa: E402
from scripts.cli.identity import ensure_identity  # noqa: E402
from scripts.cli.process_manager import (  # noqa: E402
    ManagedProcess,
    orchestrator_argv,
    orchestrator_env,
    worker_argv,
)


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN} {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}✗ {text}{Colors.END}", file=sys.stderr)


def print_info(text: str):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


class NodeManager:
    """Manage local DistribAI node."""

    def __init__(self):
        self.config_dir = Path.home() / ".distribai"
        self.config_file = self.config_dir / "desktop.json"

    def get_config(self) -> dict:
        """Get current node configuration."""
        if self.config_file.exists():
            return json.loads(self.config_file.read_text())
        return {}

    def set_config(self, config: dict):
        """Save node configuration."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file.write_text(json.dumps(config, indent=2))

    def status(self):
        """Show node status."""
        print_header("NODE STATUS")

        config = self.get_config()

        print(f"{Colors.BOLD}Configuration:{Colors.END}")
        # node_name (not nodeName) matches client/lib/identityStore.js and
        # client/routes/{settings,node,status,orchProxy}.js's desktop.json schema.
        print(f"  Node Name: {config.get('node_name') or config.get('nodeName') or 'Not set'}")
        print(f"  Node ID: {config.get('node_id', 'Not set (run `distribai node identity`)')}")
        print(f"  Org ID: {config.get('org_id', 'Not set (run `distribai node identity`)')}")
        print(f"  Region: {config.get('region', 'Not set')}")
        print(f"  CPU Limit: {config.get('cpuPercent', 50)}%")
        print(f"  GPU Limit: {config.get('gpuPercent', 50)}%")
        print(f"  RAM Limit: {config.get('ramPercent', 50)}%")

        proc_info = ManagedProcess("node").status()
        print(f"\n{Colors.BOLD}Background daemon:{Colors.END}")
        if proc_info["running"]:
            print_success(f"Running (pid {proc_info['pid']})")
        else:
            print_warning("Not running (use `distribai node start`)")

        orch_url = os.getenv("ORCHESTRATOR_ADMIN_URL", "http://127.0.0.1:8766")
        print(f"\n{Colors.BOLD}Orchestrator:{Colors.END}")
        print(f"  URL: {orch_url}")

        try:
            import urllib.request

            urllib.request.urlopen(f"{orch_url}/admin/health", timeout=2)
            print_success("Orchestrator is reachable")
        except Exception as e:
            print_error(f"Orchestrator not reachable: {e}")

    def set_resources(self, cpu: int, gpu: int, ram: int):
        """Set resource allocation percentages."""
        print_header("SET RESOURCE ALLOCATION")

        for name, value in [("CPU", cpu), ("GPU", gpu), ("RAM", ram)]:
            if not 10 <= value <= 100:
                print_error(f"{name} must be between 10 and 100")
                return

        config = self.get_config()
        config.update({"cpuPercent": cpu, "gpuPercent": gpu, "ramPercent": ram})
        self.set_config(config)

        print_success("Resource limits updated:")
        print(f"  CPU: {cpu}%")
        print(f"  GPU: {gpu}%")
        print(f"  RAM: {ram}%")

    def set_region(self, region: str):
        """Set node region."""
        print_header("SET REGION")

        valid_regions = [
            "auto",
            "us-east-1",
            "us-east-2",
            "us-west-1",
            "us-west-2",
            "eu-west-1",
            "eu-west-2",
            "eu-central-1",
            "ap-southeast-1",
            "ap-southeast-2",
            "ap-northeast-1",
            "ca-central-1",
            "sa-east-1",
        ]

        if region not in valid_regions:
            print_error(f"Invalid region: {region}")
            print_info(f"Valid regions: {', '.join(valid_regions)}")
            return

        config = self.get_config()
        config["region"] = region
        self.set_config(config)

        print_success(f"Region set to: {region}")

    def set_name(self, name: str):
        """Set node name.

        Writes ``node_name`` (not the legacy ``nodeName`` key this method used
        to write) so the value actually shows up in the dashboards and in
        ``distribai node identity`` — see client/lib/identityStore.js and
        client/routes/settings.js, which only ever read ``node_name``.
        """
        print_header("SET NODE NAME")

        config = self.get_config()
        config["node_name"] = name
        config.pop("nodeName", None)
        self.set_config(config)

        print_success(f"Node name set to: {name}")

    def show_identity(self) -> dict:
        """Print (and persist, if missing) this machine's org_id/node_id.

        Mirrors client/lib/identityStore.js so the CLI, dashboards, and
        desktop apps always agree on one identity per machine.
        """
        print_header("ORGANIZATION / NODE IDENTITY")
        config = self.get_config()
        updated, changed = ensure_identity(config)
        if changed:
            self.set_config(updated)
        print(f"  Org ID:  {Colors.BOLD}{updated['org_id']}{Colors.END}")
        print(f"  Node ID: {Colors.BOLD}{updated['node_id']}{Colors.END}")
        print(f"  Node Name: {updated.get('node_name', 'Not set')}")
        if changed:
            print_info(f"Generated and saved to {self.config_file}")
        return updated

    def start(self, orchestrator_url: str | None, worker_index: int | None) -> None:
        """Start the worker daemon in the background."""
        print_header("START NODE")
        config = self.get_config()
        node_id = config.get("node_id")
        proc = ManagedProcess("node")
        result = proc.start(worker_argv(orchestrator_url, node_id, worker_index))
        if not result.get("ok", True):
            print_error(result.get("error", "failed to start"))
            return
        print_success(f"Node daemon started (pid {result['pid']})")
        print_info(f"Logs: {result['log_file']}")

    def stop(self) -> None:
        """Stop the background worker daemon."""
        print_header("STOP NODE")
        result = ManagedProcess("node").stop()
        if not result.get("ok"):
            print_error(result.get("error", "failed to stop"))
            return
        print_success(f"Node daemon stopped (was pid {result['pid']})")

    def logs(self, lines: int) -> None:
        """Print the tail of the background worker daemon's log file."""
        print_header("NODE LOGS")
        for line in ManagedProcess("node").tail_logs(lines):
            print(line)


class JobManager:
    """Manage DistribAI jobs."""

    def __init__(self, orch_url: str | None = None):
        self.orch_url = orch_url or os.getenv("ORCHESTRATOR_ADMIN_URL", "http://127.0.0.1:8766")
        self._client = AdminAPIClient(self.orch_url)

    def _auth_headers(self) -> dict[str, str]:
        return self._client._headers()

    def _api_call(self, endpoint: str, method: str = "GET", data: dict = None) -> dict:
        """Make API call to orchestrator (kept for backward compatibility;
        delegates to the shared AdminAPIClient used by the TUI)."""
        return self._client.request(endpoint, method, data)

    def list_jobs(self, status: str | None = None):
        """List jobs."""
        print_header("JOB LIST")

        result = self._api_call("/admin/jobs")

        if "error" in result:
            print_error(f"Failed to fetch jobs: {result['error']}")
            return

        jobs = result.get("jobs", [])

        if status:
            jobs = [j for j in jobs if j.get("status") == status]

        if not jobs:
            print_info("No jobs found")
            return

        print(f"{Colors.BOLD}{'Job ID':<20} {'Status':<12} {'Model':<20} {'Steps':<8}{Colors.END}")
        print("-" * 70)

        for job in jobs[:20]:
            job_id = job.get("job_id", "N/A")[:18]
            job_status = job.get("status", "unknown")
            model = job.get("model_name", "N/A")[:18]
            steps = job.get("steps", 0)

            status_color = (
                Colors.GREEN
                if job_status == "success"
                else Colors.YELLOW
                if job_status in ["running", "assigned"]
                else Colors.RED
                if job_status in ["failed", "error"]
                else Colors.END
            )

            print(f"{job_id:<20} {status_color}{job_status:<12}{Colors.END} {model:<20} {steps:<8}")

        if len(jobs) > 20:
            print_info(f"... and {len(jobs) - 20} more jobs")

    def create_job(
        self,
        model: str,
        steps: int,
        batch_size: int = 32,
        *,
        org: str | None = None,
        job_type: str | None = None,
        priority: int | None = None,
        priority_tier: str | None = None,
        submitter_id: str | None = None,
        description: str | None = None,
        deadline_seconds: int | None = None,
        steps_per_task: int | None = None,
        learning_rate: float | None = None,
        weight_url: str | None = None,
        batch_url: str | None = None,
    ):
        """Create a new job.

        ``model``/``steps``/``batch_size`` cover the common case; every other
        parameter is optional and maps 1:1 onto ``JobCreateRequest`` fields
        (services_python/schemas.py) for parity with what a direct DB-insert
        job submission used to support.
        """
        print_header("CREATE JOB")

        body: dict[str, Any] = {"model_name": model, "steps": steps, "batch_size": batch_size}
        optional_fields = {
            "org": org,
            "job_type": job_type,
            "priority": priority,
            "priority_tier": priority_tier,
            "submitter_id": submitter_id,
            "description": description,
            "deadline_seconds": deadline_seconds,
            "steps_per_task": steps_per_task,
            "weight_blob_url": weight_url,
            "batch_blob_url": batch_url,
        }
        body.update({key: value for key, value in optional_fields.items() if value is not None})
        if learning_rate is not None:
            body["hparams"] = {"lr": learning_rate}

        result = self._api_call("/admin/jobs", "POST", body)

        if "error" in result:
            print_error(f"Failed to create job: {result['error']}")
            return

        job_id = result.get("job_id", "unknown")
        print_success(f"Job created: {job_id}")
        print(f"  Model: {model}")
        print(f"  Steps: {steps}")
        print(f"  Batch Size: {batch_size}")

    def cancel_job(self, job_id: str):
        """Cancel a job."""
        print_header("CANCEL JOB")

        result = self._api_call(f"/admin/jobs/{job_id}", "DELETE")

        if "error" in result:
            print_error(f"Failed to cancel job: {result['error']}")
            return

        print_success(f"Job {job_id} cancelled")

    def job_status(self, job_id: str) -> dict:
        """Show a single job's full status payload."""
        print_header("JOB STATUS")
        result = self._api_call(f"/admin/jobs/{job_id}")
        if "error" in result:
            print_error(f"Failed to fetch job: {result['error']}")
            return result
        job = result.get("job", result)
        for key in ("job_id", "status", "model_name", "steps", "created_at", "error_message"):
            if key in job:
                print(f"  {key}: {job[key]}")
        return job

    def watch_job(self, job_id: str, *, poll_seconds: float = 2.0, timeout_seconds: float = 600.0):
        """Poll a job until it reaches a terminal status."""
        print_header("WATCH JOB")
        terminal = {"success", "completed", "failed", "cancelled", "error"}
        deadline = time.monotonic() + timeout_seconds
        last_status = None
        while time.monotonic() < deadline:
            result = self._api_call(f"/admin/jobs/{job_id}")
            job = result.get("job", result) if isinstance(result, dict) else {}
            status = job.get("status") or result.get("error")
            if status != last_status:
                print(f"  [{job_id}] -> {status}")
                last_status = status
            if status in terminal:
                return status
            time.sleep(poll_seconds)
        print_warning("Timed out waiting for job to finish")
        return "timeout"

    @staticmethod
    def load_recipe(recipe_path: Path) -> dict:
        """Load YAML/JSON job recipe (folder, model, steps, etc.)."""
        text = recipe_path.read_text(encoding="utf-8")
        if recipe_path.suffix.lower() in (".yaml", ".yml"):
            if yaml is None:
                raise ValueError("PyYAML required for YAML recipes: pip install pyyaml")
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("recipe must be a mapping")
        return data

    @staticmethod
    def bundle_directory(folder: Path) -> tuple[bytes, str]:
        """Pack a directory into a gzip tarball; require run.py or run.ipynb."""
        if not folder.is_dir():
            raise ValueError(f"not a directory: {folder}")
        run_py = folder / "run.py"
        run_ipynb = folder / "run.ipynb"
        if not run_py.is_file() and not run_ipynb.is_file():
            raise ValueError("folder must contain run.py or run.ipynb at the top level")

        # When only a notebook is present, materialize run.py into the archive so
        # older workers without notebook mount still receive a Python entrypoint.
        generated_run_py: bytes | None = None
        if not run_py.is_file() and run_ipynb.is_file():
            _ensure_repo_on_path()
            from worker.src.sandbox.notebook_mount import ipynb_to_python

            generated_run_py = ipynb_to_python(run_ipynb.read_text(encoding="utf-8")).encode(
                "utf-8"
            )

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(folder).as_posix()
                tar.add(path, arcname=rel)
            if generated_run_py is not None:
                info = tarfile.TarInfo(name="run.py")
                info.size = len(generated_run_py)
                tar.addfile(info, io.BytesIO(generated_run_py))
        raw = buf.getvalue()
        if len(raw) > 5_000_000:
            raise ValueError("bundle exceeds 5MB limit for script_package_b64")
        return raw, hashlib.sha256(raw).hexdigest()

    def submit_folder(
        self,
        folder: Path,
        *,
        model: str = "script-job",
        steps: int = 1,
        job_type: str = "fine_tune",
    ) -> None:
        """Submit a local training folder as a script job."""
        print_header("SUBMIT SCRIPT JOB")

        try:
            package, digest = self.bundle_directory(folder)
        except ValueError as exc:
            print_error(str(exc))
            return

        from services_python.preflight import validate_script_tarball

        ok, err, _meta = validate_script_tarball(package)
        if not ok:
            print_error(f"Pre-flight failed: {err}")
            return

        pkg_b64 = base64.b64encode(package).decode("ascii")
        body = {
            "job_type": job_type,
            "base_model": model,
            "model_name": model,
            "steps": steps,
            "script_package_b64": pkg_b64,
            "hparams": {
                "execution_paradigm": "script",
                "package_sha256": digest,
            },
        }

        result = self._api_call("/admin/jobs", "POST", body)
        if "error" in result:
            print_error(f"Submit failed: {result['error']}")
            return

        job_id = result.get("job_id", "unknown")
        task_id = result.get("task_id")
        print_success(f"Job created: {job_id}")
        print(f"  Package SHA-256: {digest}")
        print(f"  Folder: {folder.resolve()}")
        if task_id:
            print(f"  Task ID: {task_id}")


class FleetViewer:
    """Operator/admin view of every node registered with an orchestrator."""

    def __init__(self, orch_url: str | None = None):
        self._client = AdminAPIClient(orch_url)

    def list_nodes(self) -> None:
        print_header("NODE FLEET")
        nodes = self._client.list_nodes()
        if not nodes:
            print_info("No nodes registered (or orchestrator unreachable)")
            return
        print(
            f"{Colors.BOLD}{'Node ID':<28} {'Status':<12} {'Online':<8} "
            f"{'Credits':<10} {'Hardware'}{Colors.END}"
        )
        print("-" * 90)
        for node in nodes:
            online = node.get("online")
            online_color = Colors.GREEN if online else Colors.YELLOW
            print(
                f"{str(node.get('node_id', 'N/A'))[:26]:<28} "
                f"{str(node.get('status', 'unknown')):<12} "
                f"{online_color}{'yes' if online else 'no':<8}{Colors.END} "
                f"{node.get('credits', 0):<10.2f} "
                f"{node.get('hardware_summary', '') or ''}"
            )


class CreditsViewer:
    """Operator/admin view of the credit ledger balances."""

    def __init__(self, orch_url: str | None = None):
        self._client = AdminAPIClient(orch_url)

    def list_credits(self) -> None:
        print_header("CREDIT BALANCES")
        credits_map = self._client.list_credits()
        if not credits_map:
            print_info("No credit balances found (or orchestrator unreachable)")
            return
        print(f"{Colors.BOLD}{'Node ID':<28} {'Balance':<12} {'Lifetime':<12}{Colors.END}")
        print("-" * 55)
        total = 0.0
        for node_id, info in credits_map.items():
            balance = float(info.get("balance") or 0)
            lifetime = float(info.get("lifetime") or 0)
            total += balance
            print(f"{node_id[:26]:<28} {balance:<12.2f} {lifetime:<12.2f}")
        print("-" * 55)
        print(f"{'Total outstanding':<28} {total:<12.2f}")


class OrchestratorController:
    """Start/stop/status/logs for the orchestrator process on this machine."""

    def start(self, grpc_port: int | None, admin_port: int | None) -> None:
        print_header("START ORCHESTRATOR")
        proc = ManagedProcess("orchestrator")
        result = proc.start(
            orchestrator_argv(grpc_port, admin_port), env=orchestrator_env(grpc_port, admin_port)
        )
        if not result.get("ok", True):
            print_error(result.get("error", "failed to start"))
            return
        print_success(f"Orchestrator started (pid {result['pid']})")
        print(f"  gRPC:  {grpc_port or 50051}")
        print(f"  Admin: {admin_port or 8766}")
        print_info(f"Logs: {result['log_file']}")

    def stop(self) -> None:
        print_header("STOP ORCHESTRATOR")
        result = ManagedProcess("orchestrator").stop()
        if not result.get("ok"):
            print_error(result.get("error", "failed to stop"))
            return
        print_success(f"Orchestrator stopped (was pid {result['pid']})")

    def status(self, orch_url: str | None) -> None:
        print_header("ORCHESTRATOR STATUS")
        info = ManagedProcess("orchestrator").status()
        if info["running"]:
            print_success(f"Process running (pid {info['pid']}, started {info.get('started_at')})")
        else:
            print_warning("Process not running locally (use `distribai orchestrator start`)")

        health = AdminAPIClient(orch_url).health()
        if "error" in health:
            print_error(f"Admin API unreachable: {health['error']}")
            return
        print_success("Admin API reachable")
        print(f"  Active nodes: {health.get('active_nodes', 0)}")
        print(f"  Queued jobs:  {health.get('queued_jobs', 0)}")
        print(f"  Running jobs: {health.get('running_jobs', 0)}")

    def logs(self, lines: int) -> None:
        print_header("ORCHESTRATOR LOGS")
        for line in ManagedProcess("orchestrator").tail_logs(lines):
            print(line)


def open_dashboard(target: str) -> None:
    """Open the contributor or operator dashboard in the default browser."""
    urls = {
        "node": os.getenv("DISTRIBAI_NODE_DASHBOARD_URL", "http://127.0.0.1:3000"),
        "orchestrator": os.getenv("DISTRIBAI_ORCH_DASHBOARD_URL", "http://127.0.0.1:3212"),
    }
    url = urls[target]
    print_info(f"Opening {url} ...")
    webbrowser.open(url)


def print_packaging_info() -> None:
    """Point each audience at the right packaging entry point.

    Deliberately a *pointer* to the existing packaging tools (specs/,
    scripts/packaging/) rather than a second implementation of them — see
    README.md "Packaging" section for the full walkthrough.
    """
    print_header("PACKAGING GUIDE")
    print(f"{Colors.BOLD}Community (contributor node binary):{Colors.END}")
    print("  python scripts/packaging/setup_wizard.py --build-only  # interactive wizard, or:")
    print("  pyinstaller specs/node-windows.spec    # Windows onedir build")
    print()
    print(f"{Colors.BOLD}Org / operator (orchestrator + dashboards):{Colors.END}")
    print("  pyinstaller specs/server-windows.spec  # Windows onedir build")
    print("  node client/orchestrator-server.js     # or run the dashboard from source")
    print()
    print(f"{Colors.BOLD}Admin (headless CLI/automation-only installs):{Colors.END}")
    print("  pip install -e .                       # registers distribai / distribai-tui")
    print("  python scripts/packaging/bundle.py cli # onefile distribai-cli.exe (+ TUI)")
    print("  python scripts/packaging/bundle.py all # admin + node + cli binaries + manifest")
    print()
    print_info("Full walkthrough: README.md 'Packaging' section, docs/guides/packaging.md")


class HealthChecker:
    """Check system health."""

    def __init__(self):
        self.orch_url = os.getenv("ORCHESTRATOR_ADMIN_URL", "http://127.0.0.1:8766")

    def full_check(self):
        """Run full health check."""
        print_header("SYSTEM HEALTH CHECK")

        checks = [
            ("Orchestrator API", self._check_orchestrator),
            ("Dashboard Server", self._check_dashboard),
            ("Node Configuration", self._check_config),
            ("Python Dependencies", self._check_python_deps),
        ]

        all_passed = True
        for name, check_func in checks:
            print(f"\n{Colors.BOLD}Checking: {name}...{Colors.END}")
            try:
                if check_func():
                    print_success(f"{name} OK")
                else:
                    print_error(f"{name} FAILED")
                    all_passed = False
            except Exception as e:
                print_error(f"{name} ERROR: {e}")
                all_passed = False

        print()
        if all_passed:
            print_success("All health checks passed!")
        else:
            print_error("Some health checks failed. See above for details.")

        return all_passed

    def _check_orchestrator(self) -> bool:
        """Check orchestrator health."""
        import urllib.request

        try:
            urllib.request.urlopen(f"{self.orch_url}/admin/health", timeout=2)
            return True
        except (urllib.error.URLError, TimeoutError):
            return False

    def _check_dashboard(self) -> bool:
        """Check dashboard server."""
        import urllib.request

        try:
            urllib.request.urlopen("http://127.0.0.1:3000", timeout=2)
            return True
        except (urllib.error.URLError, TimeoutError):
            return False

    def _check_config(self) -> bool:
        """Check node configuration exists."""
        config_file = Path.home() / ".distribai" / "desktop.json"
        return config_file.exists()

    def _check_python_deps(self) -> bool:
        """Check Python dependencies."""
        required = ["torch", "grpc", "psutil", "aiohttp"]
        missing = []

        for dep in required:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        if missing:
            print_warning(f"Missing dependencies: {', '.join(missing)}")
            return False

        return True


def _fix_console_encoding() -> None:
    """Force UTF-8 stdout/stderr so status glyphs (✓/⚠/✗/ℹ) don't crash the
    default Windows console codepage (cp1252/cp437 raise UnicodeEncodeError)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None):
    _fix_console_encoding()
    parser = argparse.ArgumentParser(
        description="DistribAI Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  distribai node status                     Show node status + identity
  distribai node start                      Start the worker daemon (background)
  distribai node set-resources 50 50 50     Set CPU/GPU/RAM to 50%
  distribai orchestrator start              Start the orchestrator (background)
  distribai orchestrator status             Process + admin API health
  distribai nodes list                      List every node registered with the orchestrator
  distribai credits list                    Show credit balances fleet-wide
  distribai job list                        List all jobs
  distribai job create distribai-small 100  Create a job
  distribai job watch <job_id>              Poll a job until it finishes
  distribai submit ./mytrainer              Submit script folder as job
  distribai submit --recipe job.yaml        Submit from human job spec file
  distribai export-weights --format onnx --out model.onnx
  distribai dashboard node                  Open the contributor GUI dashboard
  distribai package info                    Packaging entry points per audience
  distribai tui                             Launch the interactive terminal dashboard
  distribai health                          Run health check

The GUI dashboards (client/server.js, client/orchestrator-server.js) remain
the fully-featured surface; this CLI/TUI covers the operational core for
headless boxes, CI, and power users. See README.md "CLI & TUI" section.
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    node_parser = subparsers.add_parser("node", help="Manage node")
    node_subparsers = node_parser.add_subparsers(dest="node_command")

    node_subparsers.add_parser("status", help="Show node status")

    node_resources = node_subparsers.add_parser("set-resources", help="Set resource limits")
    node_resources.add_argument("cpu", type=int, help="CPU percentage (10-100)")
    node_resources.add_argument("gpu", type=int, help="GPU percentage (10-100)")
    node_resources.add_argument("ram", type=int, help="RAM percentage (10-100)")

    node_region = node_subparsers.add_parser("set-region", help="Set region")
    node_region.add_argument("region", help="Region code (e.g., us-east-1)")

    node_name = node_subparsers.add_parser("set-name", help="Set node name")
    node_name.add_argument("name", help="Node name")

    node_subparsers.add_parser("identity", help="Show/generate this machine's org_id and node_id")

    node_start = node_subparsers.add_parser("start", help="Start the worker daemon in the background")
    node_start.add_argument("--orchestrator", default=None, help="orchestrator host:port (gRPC)")
    node_start.add_argument("--worker-index", type=int, default=None, help="Multi-worker index")

    node_subparsers.add_parser("stop", help="Stop the background worker daemon")

    node_logs = node_subparsers.add_parser("logs", help="Tail the worker daemon's log file")
    node_logs.add_argument("--lines", type=int, default=60, help="Number of lines to show")

    orch_parser = subparsers.add_parser("orchestrator", help="Manage the orchestrator process")
    orch_subparsers = orch_parser.add_subparsers(dest="orchestrator_command")

    orch_start = orch_subparsers.add_parser("start", help="Start the orchestrator in the background")
    orch_start.add_argument("--grpc-port", type=int, default=None, help="gRPC port (default 50051)")
    orch_start.add_argument("--admin-port", type=int, default=None, help="Admin HTTP port (default 8766)")

    orch_subparsers.add_parser("stop", help="Stop the background orchestrator")
    orch_subparsers.add_parser("status", help="Show orchestrator process + admin API status")

    orch_logs = orch_subparsers.add_parser("logs", help="Tail the orchestrator's log file")
    orch_logs.add_argument("--lines", type=int, default=60, help="Number of lines to show")

    nodes_parser = subparsers.add_parser("nodes", help="View the node fleet (admin)")
    nodes_subparsers = nodes_parser.add_subparsers(dest="nodes_command")
    nodes_subparsers.add_parser("list", help="List every node registered with the orchestrator")

    credits_parser = subparsers.add_parser("credits", help="View credit balances (admin)")
    credits_subparsers = credits_parser.add_subparsers(dest="credits_command")
    credits_subparsers.add_parser("list", help="List credit balances for every node")

    dashboard_parser = subparsers.add_parser("dashboard", help="Open a GUI dashboard in your browser")
    dashboard_parser.add_argument(
        "target", choices=("node", "orchestrator"), help="Which dashboard to open"
    )

    package_parser = subparsers.add_parser("package", help="Show packaging instructions per audience")
    package_subparsers = package_parser.add_subparsers(dest="package_command")
    package_subparsers.add_parser("info", help="Print packaging entry points for org/community/admin")

    subparsers.add_parser("tui", help="Launch the interactive terminal dashboard")

    job_parser = subparsers.add_parser("job", help="Manage jobs")
    job_subparsers = job_parser.add_subparsers(dest="job_command")

    job_list = job_subparsers.add_parser("list", help="List jobs")
    job_list.add_argument("--status", help="Filter by status")

    job_create = job_subparsers.add_parser("create", help="Create job")
    job_create.add_argument("model", help="Model name (e.g., distribai-small)")
    job_create.add_argument("steps", type=int, help="Number of steps")
    job_create.add_argument("--batch-size", type=int, default=32, help="Batch size")
    job_create.add_argument("--org", default=None, help="Owning organization (multi-tenant)")
    job_create.add_argument("--job-type", default=None, help="fine_tune / train / rl / eval / …")
    job_create.add_argument("--priority", type=int, default=None, help="Base priority score (0-100)")
    job_create.add_argument("--priority-tier", default=None, help="Priority lane, e.g. P0/P1/P2")
    job_create.add_argument("--submitter-id", default=None, help="Submitter identifier")
    job_create.add_argument("--description", default=None, help="Optional job description")
    job_create.add_argument("--deadline-seconds", type=int, default=None, help="Per-task deadline")
    job_create.add_argument("--steps-per-task", type=int, default=None, help="Micro-task chunk size")
    job_create.add_argument("--learning-rate", type=float, default=None, help="Optimizer learning rate")
    job_create.add_argument("--weight-url", default=None, help="Weight blob URL or local path")
    job_create.add_argument("--batch-url", default=None, help="Batch/dataset blob URL or local path")

    job_cancel = job_subparsers.add_parser("cancel", help="Cancel job")
    job_cancel.add_argument("job_id", help="Job ID to cancel")

    job_status = job_subparsers.add_parser("status", help="Show one job's status")
    job_status.add_argument("job_id", help="Job ID to inspect")

    job_watch = job_subparsers.add_parser("watch", help="Poll a job until it finishes")
    job_watch.add_argument("job_id", help="Job ID to watch")
    job_watch.add_argument("--interval", type=float, default=2.0, help="Poll interval (seconds)")
    job_watch.add_argument("--timeout", type=float, default=600.0, help="Give up after N seconds")

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit a local folder (run.py or run.ipynb) as a script job",
    )
    submit_parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Path to training folder (optional when --recipe sets folder)",
    )
    submit_parser.add_argument("--model", default="script-job", help="Model name label")
    submit_parser.add_argument("--steps", type=int, default=1, help="Training steps")
    submit_parser.add_argument(
        "--orch-url",
        default=None,
        help="Orchestrator admin URL (default ORCHESTRATOR_ADMIN_URL)",
    )
    submit_parser.add_argument(
        "--recipe",
        type=Path,
        default=None,
        help="YAML/JSON recipe file (overrides folder/model/steps when set)",
    )

    export_parser = subparsers.add_parser(
        "export-weights",
        help="Export a native DistribAI model to safetensors or ONNX",
    )
    export_parser.add_argument(
        "--model",
        default="distribai-tiny",
        help="Named profile (default: distribai-tiny)",
    )
    export_parser.add_argument(
        "--architecture",
        default=None,
        help="Optional family override (dense_ffn, gru, …)",
    )
    export_parser.add_argument(
        "--format",
        choices=("safetensors", "onnx"),
        required=True,
        help="Export format",
    )
    export_parser.add_argument("--out", required=True, help="Output file path")
    export_parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Dummy sequence length for ONNX export",
    )

    subparsers.add_parser("health", help="Run health check")

    args = parser.parse_args(argv)
    _ensure_repo_on_path()

    if not args.command:
        parser.print_help()
        raise SystemExit(1)

    if args.command == "node":
        manager = NodeManager()
        if args.node_command == "status":
            manager.status()
        elif args.node_command == "set-resources":
            manager.set_resources(args.cpu, args.gpu, args.ram)
        elif args.node_command == "set-region":
            manager.set_region(args.region)
        elif args.node_command == "set-name":
            manager.set_name(args.name)
        elif args.node_command == "identity":
            manager.show_identity()
        elif args.node_command == "start":
            manager.start(args.orchestrator, args.worker_index)
        elif args.node_command == "stop":
            manager.stop()
        elif args.node_command == "logs":
            manager.logs(args.lines)
        else:
            node_parser.print_help()

    elif args.command == "orchestrator":
        controller = OrchestratorController()
        if args.orchestrator_command == "start":
            controller.start(args.grpc_port, args.admin_port)
        elif args.orchestrator_command == "stop":
            controller.stop()
        elif args.orchestrator_command == "status":
            controller.status(None)
        elif args.orchestrator_command == "logs":
            controller.logs(args.lines)
        else:
            orch_parser.print_help()

    elif args.command == "nodes":
        if args.nodes_command in (None, "list"):
            FleetViewer().list_nodes()
        else:
            nodes_parser.print_help()

    elif args.command == "credits":
        if args.credits_command in (None, "list"):
            CreditsViewer().list_credits()
        else:
            credits_parser.print_help()

    elif args.command == "dashboard":
        open_dashboard(args.target)

    elif args.command == "package":
        if args.package_command in (None, "info"):
            print_packaging_info()
        else:
            package_parser.print_help()

    elif args.command == "tui":
        # Deferred: tui.py imports this module (NodeManager/JobManager/etc.) to
        # avoid duplicating logic, so a top-level import here would be circular.
        from scripts.cli.tui import run_tui

        run_tui()

    elif args.command == "submit":
        manager = JobManager(orch_url=args.orch_url)
        folder = args.folder
        model = args.model
        steps = args.steps
        if args.recipe:
            try:
                recipe = JobManager.load_recipe(args.recipe)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                print_error(f"Invalid recipe: {exc}")
                raise SystemExit(1) from exc
            folder = Path(recipe.get("folder", folder))
            model = recipe.get("model_name", recipe.get("base_model", model))
            steps = int(recipe.get("steps", steps))
        manager.submit_folder(folder, model=model, steps=steps)

    elif args.command == "export-weights":
        from worker.src.compute.export_weights import main as export_main

        raise SystemExit(
            export_main(
                [
                    "--model",
                    args.model,
                    *(
                        ["--architecture", args.architecture]
                        if args.architecture
                        else []
                    ),
                    "--format",
                    args.format,
                    "--out",
                    args.out,
                    "--seq-len",
                    str(args.seq_len),
                ]
            )
        )

    elif args.command == "job":
        manager = JobManager()
        if args.job_command == "list":
            manager.list_jobs(args.status)
        elif args.job_command == "create":
            manager.create_job(
                args.model,
                args.steps,
                args.batch_size,
                org=args.org,
                job_type=args.job_type,
                priority=args.priority,
                priority_tier=args.priority_tier,
                submitter_id=args.submitter_id,
                description=args.description,
                deadline_seconds=args.deadline_seconds,
                steps_per_task=args.steps_per_task,
                learning_rate=args.learning_rate,
                weight_url=args.weight_url,
                batch_url=args.batch_url,
            )
        elif args.job_command == "cancel":
            manager.cancel_job(args.job_id)
        elif args.job_command == "status":
            manager.job_status(args.job_id)
        elif args.job_command == "watch":
            manager.watch_job(args.job_id, poll_seconds=args.interval, timeout_seconds=args.timeout)
        else:
            job_parser.print_help()

    elif args.command == "health":
        checker = HealthChecker()
        success = checker.full_check()
        raise SystemExit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
