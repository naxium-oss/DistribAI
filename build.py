#!/usr/bin/env python3
"""
Build script for DistribAI packaging and distribution.

This script handles building packages, creating distributions,
and preparing releases for the DistribAI system.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=check, timeout=7200
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr and result.returncode != 0:
        print(f"STDERR: {result.stderr}")
    return result


def clean_build_artifacts() -> None:
    """Clean previous build artifacts."""
    print("Cleaning build artifacts...")

    # Directories to clean
    dirs_to_clean = ["build", "dist", "*.egg-info"]
    for pattern in dirs_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                print(f"Removing directory: {path}")
                import shutil

                shutil.rmtree(path)
            else:
                print(f"Removing file: {path}")
                path.unlink()


def build_wheel() -> None:
    """Build a wheel package."""
    print("Building wheel package...")
    run_command([sys.executable, "-m", "build", "--wheel"])


def build_sdist() -> None:
    """Build a source distribution."""
    print("Building source distribution...")
    run_command([sys.executable, "-m", "build", "--sdist"])


def build_all() -> None:
    """Build both wheel and source distribution."""
    print("Building all packages...")
    run_command([sys.executable, "-m", "build"])


def install_build_deps() -> None:
    """Install build dependencies."""
    print("Installing build dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "build", "wheel", "setuptools"])


def verify_build() -> None:
    """Verify the build was successful."""
    print("Verifying build...")

    dist_dir = Path("dist")
    if not dist_dir.exists():
        raise RuntimeError("No dist/ directory found - build may have failed")

    files = list(dist_dir.glob("*"))
    if not files:
        raise RuntimeError("No files in dist/ directory - build may have failed")

    print(f"Build verification successful. Found {len(files)} files:")
    for file in files:
        print(f"  - {file.name} ({file.stat().st_size} bytes)")


def main() -> None:
    """Main build script entry point."""
    parser = argparse.ArgumentParser(description="Build DistribAI packages")
    parser.add_argument(
        "command",
        choices=["clean", "wheel", "sdist", "all", "install-deps", "verify"],
        help="Build command to run",
        default="all",
        nargs="?",
    )
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning before build")

    args = parser.parse_args()

    try:
        if args.command == "clean":
            clean_build_artifacts()
        elif args.command == "install-deps":
            install_build_deps()
        elif args.command == "wheel":
            if not args.no_clean:
                clean_build_artifacts()
            build_wheel()
            verify_build()
        elif args.command == "sdist":
            if not args.no_clean:
                clean_build_artifacts()
            build_sdist()
            verify_build()
        elif args.command == "all":
            if not args.no_clean:
                clean_build_artifacts()
            build_all()
            verify_build()
        elif args.command == "verify":
            verify_build()

        print(f"Build command '{args.command}' completed successfully!")

    except Exception as exc:
        print(f"Build failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
