"""
Mirror an allow-listed subset of the private repository into the public Grid repo.
The orchestrator and other closed-source paths stay in the private repo only.
Requires:
  PUBLIC_GRID_GIT_REMOTE — clone/push URL for the public repository
Optional:
  PUBLIC_GRID_RELEASE_ASSET_BASE — base URL written into dist/grid-manifest.json for downloads
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIRROR_PATHS = [
    "sdk/python",
    "client",
    "worker",
    "docs",
    "examples/colab",
    "examples/kaggle",
    "runtime/db/schema.sql",
    "grid.py",
    "requirements.txt",
    "requirements-worker.txt",
    "scripts/packaging/bundle.py",
    "scripts/publish/publish_public_grid.py",
    "AGENTS.md",
]
DENY_DIR_NAMES = frozenset(
    {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".pytest_cache"}
)


def _copy_tree_filtered(src: Path, dst: Path) -> None:
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in DENY_DIR_NAMES]
        rootp = Path(root)
        rel = rootp.relative_to(src)
        for fname in files:
            if fname.endswith(".pyc"):
                continue
            sfile = rootp / fname
            dfile = dst / rel / fname
            dfile.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sfile, dfile)


def _sync_paths(pub: Path) -> None:
    for rel in MIRROR_PATHS:
        src = REPO_ROOT / rel
        if not src.exists():
            print(f"skip missing source path: {rel}")
            continue
        dst = pub / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(src, dst)
        else:
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True, exist_ok=True)
            _copy_tree_filtered(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish sanitized tree to public Grid mirror")
    parser.add_argument("--push", action="store_true", default=True)
    parser.add_argument("--no-push", dest="push", action="store_false")
    args = parser.parse_args()
    remote = os.environ.get("PUBLIC_GRID_GIT_REMOTE", "").strip()
    if not remote:
        print(
            "PUBLIC_GRID_GIT_REMOTE must be set, e.g. git@github.com:YourOrg/distribai-public.git",
            file=sys.stderr,
        )
        return 1
    work = Path(tempfile.mkdtemp(prefix="grid_public_mirror_"))
    try:
        pub = work / "pub"
        subprocess.run(["git", "clone", "--depth", "1", remote, str(pub)], check=True)
        for item in pub.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        _sync_paths(pub)
        version = datetime.now(UTC).strftime("%Y%m%d.%H%M%S")
        manifest = {
            "version": version,
            "generated_at": datetime.now(UTC).isoformat(),
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
        dist_dir = pub / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        (dist_dir / "grid-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        subprocess.run(["git", "-C", str(pub), "add", "-A"], check=True)
        st = subprocess.run(
            ["git", "-C", str(pub), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        if not st.stdout.strip():
            print("No changes; public mirror already up to date.")
            return 0
        subprocess.run(
            ["git", "-C", str(pub), "commit", "-m", f"mirror: sync open components ({version})"],
            check=True,
        )
        if args.push:
            subprocess.run(["git", "-C", str(pub), "push"], check=True)
            print("Pushed public mirror.")
        else:
            print(f"Committed in staging clone (no push): {pub}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"git command failed: {e}", file=sys.stderr)
        return e.returncode or 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
