#!/usr/bin/env python3
"""
Cross-platform Python executable resolver for DistribAI.

Finds the best available Python interpreter (3.11+) in order of preference:
1. sys.executable (current interpreter)
2. python3
3. python
4. py (Windows launcher)
5. Common installation paths

Returns the path to a working Python 3.11+ interpreter.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _check_python_version(python_path: str) -> bool:
    """Check if a Python executable is version 3.11+."""
    try:
        result = subprocess.run(
            [
                python_path,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version_str = result.stdout.strip()
            major, minor = map(int, version_str.split("."))
            return major >= 3 and minor >= 11
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return False
    return False


def _get_common_python_paths() -> list[str]:
    """Get common Python installation paths for the current platform."""
    paths = []

    if sys.platform == "win32":
        # Windows common installation paths
        program_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")

        for base in [program_files, program_files_x86]:
            for version in ["313", "312", "311"]:
                python_dir = Path(base) / "Python" / version
                if python_dir.exists():
                    paths.append(str(python_dir / "python.exe"))
                    paths.append(str(python_dir / "pythonw.exe"))

        # Also check in user's local AppData
        local_app_data = os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        programs_dir = Path(local_app_data) / "Programs" / "Python"
        if programs_dir.exists():
            for python_dir in programs_dir.glob("Python*"):
                python_exe = python_dir / "python.exe"
                if python_exe.exists():
                    paths.append(str(python_exe))

    elif sys.platform == "darwin":
        # macOS common installation paths
        for version in ["3.13", "3.12", "3.11"]:
            paths.extend(
                [
                    f"/usr/bin/python{version}",
                    f"/usr/local/bin/python{version}",
                    f"/opt/homebrew/bin/python{version}",
                    f"/Library/Frameworks/Python.framework/Versions/{version}/bin/python{version}",
                ]
            )

    else:  # Linux and other Unix-like
        for version in ["3.13", "3.12", "3.11"]:
            paths.extend(
                [
                    f"/usr/bin/python{version}",
                    f"/usr/local/bin/python{version}",
                    f"/opt/python/{version}/bin/python",
                ]
            )

    return paths


def find_python() -> str:
    """
    Find the best available Python 3.11+ interpreter.

    Returns:
        Path to a working Python executable

    Raises:
        RuntimeError: If no suitable Python is found
    """
    current_exe = sys.executable
    if current_exe and Path(current_exe).exists():
        return current_exe

    candidates = []

    if sys.platform == "win32":
        candidates.append("py")

    candidates.extend(["python3", "python"])

    for candidate in candidates:
        python_path = shutil.which(candidate)
        if python_path and _check_python_version(python_path):
            return python_path

    for path in _get_common_python_paths():
        if Path(path).exists() and _check_python_version(path):
            return path

    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        return sys.executable

    raise RuntimeError(
        "Python 3.11+ not found. Please install Python 3.11 or newer from https://python.org"
    )


def get_python_command() -> list[str]:
    """
    Get the command list to execute Python.

    Returns:
        List of command components to execute Python
    """
    python_path = find_python()

    # For Windows py launcher, we need to specify the version
    if sys.platform == "win32" and python_path.endswith("py.exe"):
        return [python_path, "-3.11"]

    return [python_path]


if __name__ == "__main__":
    try:
        python_path = find_python()
        print(python_path)
        raise SystemExit(0)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
