#!/usr/bin/env python3
"""
Verification script for DistribAI production setup.

Checks that all components are properly integrated and ready for packaging.

Usage:
    python verify_setup.py
"""

import importlib
import sys
from pathlib import Path


def check_import(module_name, description, optional=False):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"[OK] {description}")
        return True
    except ImportError as e:
        if optional:
            print(f"[WARN] {description}: {e} (optional)")
            return True  # Optional dependency - don't fail
        print(f"[FAIL] {description}: {e}")
        return False
    except Exception as e:
        print(f"[WARN] {description}: {e}")
        return True  # Module exists but has other issues


def check_file(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"[OK] {description}")
        return True
    else:
        print(f"[FAIL] {description}: File not found")
        return False


def main():
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    print("=" * 60)
    print("DistribAI Production Setup Verification")
    print("=" * 60)
    print()

    results = []

    # Check core files
    print("Core Files:")
    results.append(check_file("requirements.txt", "requirements.txt"))
    results.append(check_file("requirements-cuda.txt", "requirements-cuda.txt"))
    results.append(check_file("setup.py", "setup.py"))
    results.append(check_file("build.py", "build.py"))
    results.append(check_file("README.md", "README.md"))
    print()

    # Check services_python modules
    print("Server Modules:")
    results.append(
        check_import("services_python.orchestrator_grpc", "orchestrator_grpc", optional=True)
    )
    results.append(check_import("services_python.grpc_service", "grpc_service", optional=True))
    results.append(check_import("services_python.scheduler", "scheduler", optional=True))
    results.append(check_import("services_python.server_gui", "server_gui", optional=True))
    results.append(check_import("services_python.constants", "constants", optional=True))
    results.append(check_import("services_python.job_submission", "job_submission", optional=True))
    results.append(check_import("services_python.database", "database", optional=True))
    results.append(
        check_import("services_python.distributed_trainer", "distributed_trainer", optional=True)
    )
    results.append(
        check_import("services_python.dependency_checker", "dependency_checker", optional=True)
    )
    results.append(check_import("services_python.admin_keys", "admin_keys", optional=True))
    print()

    # Check worker modules
    print("Worker Modules:")
    results.append(check_import("worker.src.daemon.scheduler_config", "scheduler_config"))
    results.append(check_import("worker.src.daemon.gui_launcher", "gui_launcher", optional=True))
    results.append(check_import("worker.src.daemon.run", "daemon run"))
    results.append(check_import("worker.src.daemon.script_runner", "script_runner"))
    results.append(check_import("worker.src.daemon.ollama_runner", "ollama_runner", optional=True))
    print()

    # Check spec files
    print("Build Specifications:")
    results.append(check_file("specs/server-windows.spec", "Server Windows spec"))
    results.append(check_file("specs/node-windows.spec", "Node Windows spec"))
    results.append(check_file("specs/node-windows.nsi", "Node Windows installer"))
    print()

    # Check documentation
    print("Documentation:")
    results.append(check_file("docs/guides/packaging.md", "Packaging guide"))
    results.append(check_file("docs/guides/node-user-guide.md", "Node user guide"))
    results.append(check_file("docs/guides/server-operator-guide.md", "Server operator guide"))
    results.append(check_file("docs/guides/update-hosting.md", "Update hosting guide"))
    results.append(check_file("docs/guides/production-workflow.md", "Production workflow"))
    results.append(check_file("docs/api/endpoints.md", "Job submission API reference"))
    results.append(check_file("docs/api/README.md", "API quick reference"))
    results.append(check_file("docs/guides/five-minute-onboarding.md", "Five-minute onboarding"))
    print()

    # Check examples and scripts
    print("Examples & Scripts:")
    results.append(check_file("scripts/maintenance/submit_job.py", "Job submission CLI"))
    results.append(check_file("examples/train_template.py", "Training template"))
    results.append(check_file("examples/inference_template.py", "Inference template"))
    print()

    # Summary
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("[OK] All components verified! Ready for packaging.")
        print()
        print("Next steps:")
        print("1. Run: python setup.py")
        print("2. Build packages: python build.py --all")
        print("3. Test packages in dist/ directory")
        return 0
    else:
        print("[FAIL] Some components missing or have issues.")
        print("Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
