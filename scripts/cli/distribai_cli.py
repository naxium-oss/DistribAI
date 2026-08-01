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
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_repo_on_path() -> None:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


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
        print(f"  Node Name: {config.get('nodeName', 'Not set')}")
        print(f"  Region: {config.get('region', 'Not set')}")
        print(f"  CPU Limit: {config.get('cpuPercent', 50)}%")
        print(f"  GPU Limit: {config.get('gpuPercent', 50)}%")
        print(f"  RAM Limit: {config.get('ramPercent', 50)}%")

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
        """Set node name."""
        print_header("SET NODE NAME")

        config = self.get_config()
        config["nodeName"] = name
        self.set_config(config)

        print_success(f"Node name set to: {name}")


class JobManager:
    """Manage DistribAI jobs."""

    def __init__(self, orch_url: str | None = None):
        self.orch_url = orch_url or os.getenv("ORCHESTRATOR_ADMIN_URL", "http://127.0.0.1:8766")

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        secret = os.getenv("DISTRIBAI_ADMIN_SECRET", "").strip()
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        return headers

    def _api_call(self, endpoint: str, method: str = "GET", data: dict = None) -> dict:
        """Make API call to orchestrator."""
        import urllib.error
        import urllib.request

        url = f"{self.orch_url}{endpoint}"

        try:
            if method == "GET":
                req = urllib.request.Request(url, headers=self._auth_headers())
                with urllib.request.urlopen(req, timeout=30) as response:
                    return json.loads(response.read().decode())
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode() if data else None,
                    headers=self._auth_headers(),
                    method=method,
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}

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

    def create_job(self, model: str, steps: int, batch_size: int = 32):
        """Create a new job."""
        print_header("CREATE JOB")

        result = self._api_call(
            "/admin/jobs", "POST", {"model_name": model, "steps": steps, "batch_size": batch_size}
        )

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


def main():
    parser = argparse.ArgumentParser(
        description="DistribAI Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  distribai_cli.py node status              Show node status
  distribai_cli.py node set-resources 50 50 50  Set CPU/GPU/RAM to 50%%
  distribai_cli.py node set-region us-east-1    Set region
  distribai_cli.py job list                 List all jobs
  distribai_cli.py job create distribai-small 100  Create a job
  distribai_cli.py submit ./mytrainer       Submit script folder as job
  distribai_cli.py submit --recipe job.yaml Submit from human job spec file
  distribai_cli.py export-weights --format onnx --out model.onnx
  distribai_cli.py health                   Run health check
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

    job_parser = subparsers.add_parser("job", help="Manage jobs")
    job_subparsers = job_parser.add_subparsers(dest="job_command")

    job_list = job_subparsers.add_parser("list", help="List jobs")
    job_list.add_argument("--status", help="Filter by status")

    job_create = job_subparsers.add_parser("create", help="Create job")
    job_create.add_argument("model", help="Model name (e.g., distribai-small)")
    job_create.add_argument("steps", type=int, help="Number of steps")
    job_create.add_argument("--batch-size", type=int, default=32, help="Batch size")

    job_cancel = job_subparsers.add_parser("cancel", help="Cancel job")
    job_cancel.add_argument("job_id", help="Job ID to cancel")

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

    args = parser.parse_args()
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
        else:
            node_parser.print_help()

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
            manager.create_job(args.model, args.steps, args.batch_size)
        elif args.job_command == "cancel":
            manager.cancel_job(args.job_id)
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
