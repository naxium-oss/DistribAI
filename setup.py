#!/usr/bin/env python3
"""
DistribAI packaging wizard (interactive PyInstaller driver).

Prefer day-to-day development via `requirements.txt`, `npm install`, and the
dual-dashboard harness documented in README.md.

What this script does:

- Validates Python >= 3.11 and GPU presence when CUDA is selected.
- Installs dependencies from `requirements.txt` or `requirements-cuda.txt`.
- Invokes PyInstaller for ad-hoc server/node bundles (see `specs/*.spec` for
  canonical spec files maintained by the project).

Usage:
    python setup.py [--quick] [--build-only] [--cuda | --cpu]
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from scripts.ci.find_python import find_python


# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def clear():
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    """Print welcome banner."""
    print(
        f"""
{Colors.CYAN}{Colors.BOLD}
====================================================================
=                                                                  =
=                    DistribAI packaging wizard                     =
=                                                                  =
=               PyInstaller bundles (experimental)               =
====================================================================
{Colors.END}
    """
    )


def print_step(step_num: int, total: int, title: str):
    """Print step header."""
    print(f"\n{Colors.BLUE}[{step_num}/{total}] {Colors.BOLD}{title}{Colors.END}\n")


def ask_yn(prompt: str, default: bool = True) -> bool:
    """Ask yes/no question."""
    suffix = " [Y/n] " if default else " [y/N] "
    response = input(f"{Colors.YELLOW}?{Colors.END} {prompt}{suffix}").strip().lower()
    if not response:
        return default
    return response in ("y", "yes")


def ask_choice(prompt: str, options: list[str], default: int = 0) -> int:
    """Ask user to choose from options."""
    print(f"\n{Colors.CYAN}{prompt}{Colors.END}")
    for i, opt in enumerate(options, 1):
        marker = "->" if i - 1 == default else "  "
        print(f"  {marker} {i}. {opt}")
    while True:
        try:
            response = input(f"\nChoice (1-{len(options)}, default {default + 1}): ").strip()
            if not response:
                return default
            choice = int(response) - 1
            if 0 <= choice < len(options):
                return choice
            print(f"{Colors.RED}Invalid choice. Please enter 1-{len(options)}.{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}Please enter a number.{Colors.END}")


def check_python_version() -> bool:
    """Check Python version is 3.11+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(
            f"{Colors.RED}ERROR: Python 3.11+ required, found {version.major}.{version.minor}{Colors.END}"
        )
        print("Please upgrade Python: https://python.org/downloads")
        return False
    print(
        f"{Colors.GREEN}[OK] Python {version.major}.{version.minor}.{version.micro} detected{Colors.END}"
    )
    return True


def check_cuda() -> tuple[bool, str | None]:
    """Check if CUDA is available and return version."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False, timeout=10
        )
        if result.returncode == 0 and "CUDA Version" in result.stdout:
            # Extract CUDA version
            for line in result.stdout.split("\n"):
                if "CUDA Version:" in line:
                    version = line.split("CUDA Version:")[1].split()[0]
                    return True, version
            return True, "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False, None


def check_gpu() -> tuple[bool, str | None]:
    """Check GPU model."""
    cuda_available, cuda_version = check_cuda()
    if not cuda_available:
        return False, None

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
    except ImportError:
        pass

    return cuda_available, "NVIDIA GPU (unknown model)"


def install_requirements(cuda: bool = False) -> bool:
    """Install Python dependencies."""
    req_file = "requirements-cuda.txt" if cuda else "requirements.txt"

    print(f"{Colors.CYAN}Installing dependencies from {req_file}...{Colors.END}")
    print("This may take several minutes...\n")

    python_path = find_python()
    cmd = [python_path, "-m", "pip", "install", "-r", req_file]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode == 0:
            print(f"{Colors.GREEN}[OK] Dependencies installed successfully{Colors.END}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}[X] Failed to install dependencies: {e}{Colors.END}")
        return False

    return False


def verify_torch(cuda: bool) -> bool:
    """Verify PyTorch installation."""
    try:
        import torch

        print(f"{Colors.GREEN}[OK] PyTorch {torch.__version__} installed{Colors.END}")

        if cuda:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                print(f"{Colors.GREEN}  [OK] CUDA available: {cuda_version}{Colors.END}")
                print(f"{Colors.GREEN}  [OK] GPU: {gpu_name}{Colors.END}")
            else:
                print(
                    f"{Colors.YELLOW}[!] CUDA PyTorch installed but CUDA not available at runtime{Colors.END}"
                )
                print(
                    f"{Colors.YELLOW}  This is OK for building, but runtime will use CPU{Colors.END}"
                )
        else:
            print(f"{Colors.GREEN}  [OK] CPU-only mode{Colors.END}")

        return True
    except ImportError:
        print(f"{Colors.RED}[X] PyTorch not properly installed{Colors.END}")
        return False


def build_server_package(platform_name: str, onefile: bool = False) -> bool:
    """Build Server package using PyInstaller."""
    print(f"\n{Colors.CYAN}Building Server package for {platform_name}...{Colors.END}")

    script_path = Path("services_python/server_gui.py")
    if not script_path.exists():
        print(f"{Colors.RED}[X] server_gui.py not found. Creating minimal launcher...{Colors.END}")
        create_server_gui()

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        f"DistribAI-Server-{platform_name}",
        "--clean",
        "--noconfirm",
        "--hidden-import",
        "services_python.orchestrator_grpc",
        "--hidden-import",
        "worker.src.daemon.byzantine_detector",
        "--hidden-import",
        "grpc",
        "--hidden-import",
        "aiohttp",
        "--hidden-import",
        "torch",
        "--collect-all",
        "torch",
        "--add-data",
        "worker/src/dashboard/static:static",
        str(script_path),
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode == 0:
            print(f"{Colors.GREEN}[OK] Server package built successfully{Colors.END}")
            print(f"{Colors.CYAN}  Output: dist/DistribAI-Server-{platform_name}{Colors.END}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}[X] Build failed: {e}{Colors.END}")
        return False

    return False


def build_node_package(platform_name: str, onefile: bool = False) -> bool:
    """Build Node package using PyInstaller."""
    print(f"\n{Colors.CYAN}Building Node package for {platform_name}...{Colors.END}")

    script_path = Path("worker/src/daemon/gui_launcher.py")
    if not script_path.exists():
        print(f"{Colors.RED}[X] gui_launcher.py not found. Creating...{Colors.END}")
        create_node_gui()

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        f"DistribAI-Node-{platform_name}",
        "--clean",
        "--noconfirm",
        "--hidden-import",
        "worker.src.daemon.run",
        "--hidden-import",
        "worker.src.daemon.scheduler_config",
        "--hidden-import",
        "grpc",
        "--hidden-import",
        "torch",
        "--collect-all",
        "torch",
        "--add-data",
        "worker/src/dashboard/static:static",
        str(script_path),
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        if result.returncode == 0:
            print(f"{Colors.GREEN}[OK] Node package built successfully{Colors.END}")
            print(f"{Colors.CYAN}  Output: dist/DistribAI-Node-{platform_name}{Colors.END}")
            return True
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}[X] Build failed: {e}{Colors.END}")
        return False

    return False


def create_server_gui():
    """Create minimal server GUI if not exists."""
    content = '''
"""Server GUI launcher using PyWebView."""
import webview
import sys
from pathlib import Path
from scripts.ci.find_python import find_python

def main():
    """Launch server and open GUI window."""
    # Start server in background
    import asyncio
    from services_python.orchestrator_grpc import serve

    # Launch webview
    window = webview.create_window(
        'DistribAI Server',
        'worker/src/dashboard/static/node/index.html',
        width=1400,
        height=900,
        min_size=(800, 600)
    )
    webview.start(debug=False)

if __name__ == '__main__':
    main()
'''
    path = Path("services_python/server_gui.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def create_node_gui():
    """Create minimal node GUI if not exists."""
    content = '''
"""Node GUI launcher using PyWebView."""
import webview
import sys
from pathlib import Path
from scripts.ci.find_python import find_python

def main():
    """Launch node and open GUI window."""
    window = webview.create_window(
        'DistribAI Node',
        'worker/src/dashboard/static/node/index.html',
        width=1200,
        height=800,
        min_size=(600, 400)
    )
    webview.start(debug=False)

if __name__ == '__main__':
    main()
'''
    path = Path("worker/src/daemon/gui_launcher.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main():
    """Main setup wizard."""
    parser = argparse.ArgumentParser(
        description="DistribAI Packaging Setup Wizard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python setup.py              # Full interactive setup
    python setup.py --quick      # Quick mode (default options)
    python setup.py --build-only # Skip to build step
        """,
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode with defaults")
    parser.add_argument("--build-only", action="store_true", help="Skip to build step")
    parser.add_argument("--cuda", action="store_true", help="Force CUDA install")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only install")
    args = parser.parse_args()

    clear()
    banner()

    # Step 1: Environment Check
    print_step(1, 4, "Environment Check")

    if not check_python_version():
        sys.exit(1)

    # Detect CUDA
    cuda_available, cuda_version = check_cuda()
    if cuda_available:
        print(f"{Colors.GREEN}[OK] CUDA {cuda_version} detected{Colors.END}")
        gpu_available, gpu_name = check_gpu()
        if gpu_available:
            print(f"{Colors.GREEN}[OK] GPU: {gpu_name}{Colors.END}")
    else:
        print(f"{Colors.YELLOW}[!] No CUDA detected (CPU-only mode){Colors.END}")

    # Determine CUDA mode
    use_cuda = cuda_available
    if args.cuda:
        use_cuda = True
    elif args.cpu:
        use_cuda = False
    elif not args.quick and not args.build_only:
        if cuda_available:
            use_cuda = ask_yn("Install with CUDA support?", default=True)
        else:
            print(
                f"\n{Colors.YELLOW}CUDA not detected. You can still build CPU-only packages.{Colors.END}"
            )
            use_cuda = False

    # Step 2: Install Dependencies
    if not args.build_only:
        print_step(2, 4, "Install Dependencies")

        print(f"Mode: {Colors.CYAN}{'CUDA' if use_cuda else 'CPU-only'}{Colors.END}\n")

        if ask_yn("Install Python dependencies now?", default=True):
            if not install_requirements(cuda=use_cuda):
                print(f"\n{Colors.RED}[X] Dependency installation failed{Colors.END}")
                sys.exit(1)

            # Verify installation
            if not verify_torch(cuda=use_cuda):
                print(
                    f"\n{Colors.YELLOW}[!] PyTorch verification had issues, but continuing...{Colors.END}"
                )

    # Step 3: Build Configuration
    print_step(3, 4, "Build Configuration")

    if not args.quick:
        print("Build Options:\n")

        # Package format
        format_choice = ask_choice(
            "Package format:", ["Single-folder (recommended)", "Single-file executable"], default=0
        )
        onefile = format_choice == 1

        # Platforms to build
        print(f"\n{Colors.CYAN}Select platforms to build:{Colors.END}")
        build_server = ask_yn("Build Server package?", default=True)
        build_node = ask_yn("Build Node package?", default=True)
    else:
        onefile = False
        build_server = True
        build_node = True

    # Step 4: Build Packages
    print_step(4, 4, "Build Packages")

    platform_name = platform.system().lower()

    success_count = 0
    total_count = 0

    if build_server:
        total_count += 1
        if build_server_package(platform_name, onefile):
            success_count += 1

    if build_node:
        total_count += 1
        if build_node_package(platform_name, onefile):
            success_count += 1

    # Summary
    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.END}")
    print(f"\n{Colors.BOLD}Build Summary:{Colors.END}\n")
    print(f"  Platforms built: {success_count}/{total_count}")

    if success_count == total_count:
        print(f"\n{Colors.GREEN}[OK] All packages built successfully!{Colors.END}")
        print(f"\n{Colors.CYAN}Output location: {Path('dist').absolute()}{Colors.END}")
        print(f"\n{Colors.BOLD}Next steps:{Colors.END}")
        print("  1. Smoke-test artefacts in dist/")
        print("  2. Prefer tracked PyInstaller specs under specs/ for repeatable builds")
        print("  3. Publishing binaries is project-specific — see docs/guides/packaging.md")
    else:
        print(f"\n{Colors.YELLOW}[!] Some builds failed. Check output above.{Colors.END}")

    print(f"\n{Colors.CYAN}{'=' * 60}{Colors.END}\n")


if __name__ == "__main__":
    main()
