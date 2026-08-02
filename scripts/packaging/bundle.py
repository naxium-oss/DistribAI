import json
import os
import subprocess
import sys
from pathlib import Path


def build_admin():
    print("Building DistribAI Admin...")
    entry = "services_python/orchestrator_grpc.py"
    exe_name = os.environ.get("PYINSTALLER_ADMIN_NAME", "distribai-admin")
    cmd = [
        "pyinstaller",
        "--name",
        exe_name,
        "--onefile",
        "--add-data",
        f"worker/src/dashboard/static{os.pathsep}dashboard/static",
        "--add-data",
        f"runtime/db/schema.sql{os.pathsep}runtime/db",
        "--hidden-import",
        "aiohttp",
        "--hidden-import",
        "grpc",
        "--hidden-import",
        "jwt",
        "--hidden-import",
        "markdown",
        entry,
    ]
    subprocess.run(cmd, check=True, timeout=7200)


def build_node():
    print("Building DistribAI Node...")
    entry = "worker/src/daemon/run.py"
    exe_name = os.environ.get("PYINSTALLER_NODE_NAME", "distribai-node")
    # Mirrors specs/node-windows.spec's hiddenimports (minus pywebview/clr_loader,
    # which run.py's headless daemon never imports) — a onefile PyInstaller build
    # only sees modules Python actually `import`s at analysis time, so anything
    # missing here reproduces the same ModuleNotFoundError class of bug that
    # `worker.src.daemon.job_executor` (now `executor`) caused in the .spec file.
    hidden_imports = [
        "aiohttp",
        "grpc",
        "grpc.aio",
        "psutil",
        "torch",
        "torch.cuda",
        "torch.nn",
        "torch.nn.functional",
        "torch.optim",
        "numpy",
        "worker.src.daemon.run",
        "worker.src.daemon.scheduler_config",
        "worker.src.daemon.executor",
        "worker.src.daemon.byzantine_detector",
        "worker.src.daemon.credit_ledger",
        "worker.src.daemon.voting_system",
        "worker.src.daemon.gradient_compression",
        "worker.src.daemon.ml_core",
        "worker.src.distribai_proto",
        "worker.src.compute.distribai_models",
        "worker.src.compute.external_arch",
    ]
    cmd = ["pyinstaller", "--name", exe_name, "--onefile"]
    cmd += ["--add-data", f"worker/src/dashboard/static{os.pathsep}dashboard/static"]
    for module in hidden_imports:
        cmd += ["--hidden-import", module]
    cmd.append(entry)
    subprocess.run(cmd, check=True, timeout=7200)


def build_cli():
    """Onefile ``distribai`` binary for admins without a Python install.

    Bundles the flat CLI and the Textual TUI (``distribai-cli tui``) into one
    executable; this is the "admin" packaging profile referenced in
    README.md's Packaging section and `distribai package info`.
    """
    print("Building DistribAI CLI...")
    entry = "scripts/cli/distribai_cli.py"
    exe_name = os.environ.get("PYINSTALLER_CLI_NAME", "distribai-cli")
    cmd = [
        "pyinstaller",
        "--name",
        exe_name,
        "--onefile",
        "--console",
        "--hidden-import",
        "scripts.cli.api_client",
        "--hidden-import",
        "scripts.cli.identity",
        "--hidden-import",
        "scripts.cli.process_manager",
        "--hidden-import",
        "scripts.cli.tui",
        "--hidden-import",
        "textual",
        "--hidden-import",
        "rich",
        entry,
    ]
    subprocess.run(cmd, check=True, timeout=7200)


def write_manifest():
    dist_dir = Path(__file__).resolve().parents[2] / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    man = {
        "version": os.environ.get("GRID_RELEASE_VERSION", "dev"),
        "assets": {
            "windows": os.environ.get("GRID_ASSET_NAME_WIN", "distribai-admin.exe"),
            "macos": os.environ.get("GRID_ASSET_NAME_MACOS", "distribai-admin-macos"),
            "linux": os.environ.get("GRID_ASSET_NAME_LINUX", "distribai-admin-linux"),
        },
        "download_base_url": os.environ.get(
            "PUBLIC_GRID_RELEASE_ASSET_BASE",
            "https://github.com/naxium-oss/DistribAI/releases/latest/download",
        ),
    }
    (dist_dir / "grid-manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {dist_dir / 'grid-manifest.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/packaging/bundle.py [admin|node|cli|all|manifest]")
        sys.exit(1)
    target = sys.argv[1]
    if target in ["admin", "all"]:
        build_admin()
    if target in ["node", "all"]:
        build_node()
    if target in ["cli", "all"]:
        build_cli()
    if target in ["manifest", "all"]:
        write_manifest()
