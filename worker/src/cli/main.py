"""
CLI for DistribAI Management

Provides command-line interface for managing nodes, jobs, and
interacting with the DistribAI database.
"""

import argparse
import json
import secrets
import time
from pathlib import Path

from tabulate import tabulate

try:
    from services_python.db_manager import DBManager
except ImportError:
    from db_manager import DBManager


def get_db() -> DBManager:
    """
    Get database manager instance.

    Returns:
        DBManager configured with project database path

    Example:
        >>> db = get_db()
        >>> nodes = db.get_all_nodes()
    """
    repo_root = Path(__file__).resolve().parents[3]
    db_path = repo_root / "runtime" / "db" / "distribai.db"
    schema_path = db_path.parent / "schema.sql"
    return DBManager(str(db_path), str(schema_path))


def cmd_nodes(args) -> None:
    """
    List all active nodes in the grid.

    Args:
        args: Command line arguments (unused)

    Example:
        >>> # distribai nodes
        >>> # Displays table of all nodes with status and hardware
    """
    db = get_db()
    nodes = db.get_all_nodes()
    if not nodes:
        print("No active nodes found.")
        return
    headers = ["Node ID", "Status", "Last Heartbeat", "Details"]
    table = []
    for node in nodes:
        hw = node.get("hardware", {})
        details = f"{hw.get('gpu_model', 'CPU')} | {hw.get('ram_gb', 0)}GB"
        table.append([node["node_id"], node["status"], node["last_heartbeat_ts"], details])
    print(tabulate(table, headers=headers, tablefmt="grid"))


def cmd_jobs(args) -> None:
    """
    List all jobs in the system.

    Args:
        args: Command line arguments (unused)

    Example:
        >>> # distribai jobs
        >>> # Displays table of all jobs with status and votes
    """
    db = get_db()
    jobs = db.get_all_jobs()
    if not jobs:
        print("No jobs found.")
        return
    headers = ["Job ID", "Model", "Status", "Priority", "Votes", "Created At"]
    table = []
    for job in jobs:
        table.append(
            [
                job["job_id"],
                job["model_name"],
                job["status"],
                job["priority"],
                job["total_votes"],
                job["created_at"],
            ]
        )
    print(tabulate(table, headers=headers, tablefmt="grid"))


def cmd_submit(args) -> None:
    """
    Submit a new training job to the grid.

    Args:
        args: Command arguments with model, job_type, steps, etc.

    Example:
        >>> # distribai submit --model llama-7b --steps 1000
    """
    db = get_db()
    job_id = f"job_{secrets.token_urlsafe(12)}"
    job = {
        "job_id": job_id,
        "model_name": args.model,
        "job_type": args.job_type,
        "base_model": args.model,
        "dataset_ref": args.batch_url,
        "description": args.description,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "priority": args.priority,
        "priority_tier": args.priority_tier,
        "submitter_id": args.submitter_id,
        "org": args.org,
        "status": "queued",
        "created_at": int(time.time()),
        "batch_blob_url": args.batch_url,
        "weight_blob_url": args.weight_url,
        "hparams": {"lr": args.learning_rate, "batch_size": args.batch_size},
        "deadline_seconds": args.deadline_seconds,
    }
    task_ids = db.insert_job_with_tasks(job, steps_per_task=args.steps_per_task)
    print(json.dumps({"job_id": job_id, "task_ids": task_ids}, indent=2))


def main() -> None:
    """
    Main entry point for CLI.

    Parses command-line arguments and dispatches to appropriate handlers.

    Example:
        >>> # python -m worker.src.cli.main nodes
        >>> # python -m worker.src.cli.main submit --model llama-7b
    """
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
        cmd_nodes(args)
    elif args.command == "jobs":
        cmd_jobs(args)
    elif args.command == "submit":
        cmd_submit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
