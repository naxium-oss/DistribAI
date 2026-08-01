"""Legacy entry point for the ``distribai`` console script.

This used to be an independent CLI that queried SQLite directly (bypassing
the admin API's auth/validation) and hard-required ``tabulate``, a package
never declared in requirements.txt/pyproject.toml — so the registered
``distribai`` script crashed with ``ModuleNotFoundError`` on any clean
install. It now delegates to the consolidated, actively maintained CLI in
``scripts/cli/distribai_cli.py`` (also home to the ``distribai-tui``
terminal dashboard), translating this module's older flag names so existing
scripts/muscle-memory keep working.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.cli.distribai_cli import main as _consolidated_main  # noqa: E402


def _translate_submit(args: argparse.Namespace) -> list[str]:
    argv = ["job", "create", args.model, str(args.steps), "--batch-size", str(args.batch_size)]
    argv += ["--job-type", args.job_type, "--org", args.org]
    argv += ["--priority", str(args.priority), "--priority-tier", args.priority_tier]
    argv += ["--submitter-id", args.submitter_id]
    argv += ["--deadline-seconds", str(args.deadline_seconds)]
    argv += ["--steps-per-task", str(args.steps_per_task)]
    argv += ["--learning-rate", str(args.learning_rate)]
    if args.description:
        argv += ["--description", args.description]
    if args.weight_url:
        argv += ["--weight-url", args.weight_url]
    if args.batch_url:
        argv += ["--batch-url", args.batch_url]
    return argv


def main() -> None:
    """Parse this module's legacy flags, then dispatch to the real CLI."""
    parser = argparse.ArgumentParser(prog="distribai", description="DistribAI Grid Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    subparsers.add_parser("nodes", help="List active nodes")
    subparsers.add_parser("jobs", help="List active jobs")
    submit = subparsers.add_parser("submit", help="Submit a new training job")
    submit.add_argument("--model", default="distribai-small", help="Model name")
    submit.add_argument("--job-type", default="fine_tune", help="Job type")
    submit.add_argument("--description", default="", help="Optional job description")
    submit.add_argument("--submitter-id", default="cli-user", help="Submitter identifier")
    submit.add_argument("--org", default="community", help="Owning organization")
    submit.add_argument("--priority", type=int, default=0, help="Base priority score")
    submit.add_argument("--priority-tier", default="P1", help="Priority tier, e.g. P0/P1/P2")
    submit.add_argument("--steps", type=int, default=100, help="Total training steps")
    submit.add_argument(
        "--steps-per-task", type=int, default=25, help="Micro-task decomposition size"
    )
    submit.add_argument("--batch-size", type=int, default=32, help="Batch size")
    submit.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    submit.add_argument("--deadline-seconds", type=int, default=600, help="Per-task deadline")
    submit.add_argument("--weight-url", default="", help="Weight blob URL or local path")
    submit.add_argument("--batch-url", default="", help="Batch blob URL or local path")
    args = parser.parse_args()

    if args.command == "nodes":
        _consolidated_main(["nodes", "list"])
    elif args.command == "jobs":
        _consolidated_main(["job", "list"])
    elif args.command == "submit":
        _consolidated_main(_translate_submit(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
