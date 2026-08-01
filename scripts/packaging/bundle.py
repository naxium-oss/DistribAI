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
    cmd = [
        "pyinstaller",
        "--name",
        exe_name,
        "--onefile",
        "--add-data",
        f"worker/src/dashboard/static{os.pathsep}dashboard/static",
        "--hidden-import",
        "aiohttp",
        "--hidden-import",
        "grpc",
        "--hidden-import",
        "psutil",
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
        print("Usage: python scripts/packaging/bundle.py [admin|node|all|manifest]")
        sys.exit(1)
    target = sys.argv[1]
    if target in ["admin", "all"]:
        build_admin()
    if target in ["node", "all"]:
        build_node()
    if target == "manifest":
        write_manifest()
