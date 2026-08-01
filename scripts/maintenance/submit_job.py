#!/usr/bin/env python3
"""
Job submission helper script for organizations.

Usage:
    python submit_job.py --org my-org --name "Training Job" --type train --script train.py

Environment:
    DISTRIBAI_SERVER_URL - Server URL (default: http://localhost:8766)
    DISTRIBAI_API_KEY - API key for authentication
"""

import argparse
import os
import sys
from pathlib import Path

import requests


def submit_job(
    server_url: str,
    api_key: str,
    org_id: str,
    name: str,
    job_type: str,
    priority: str,
    script_path: str,
    base_model: str,
    dataset: str,
    steps: int,
    vram_gb: float,
    requirements: list[str],
) -> dict:
    """Submit a job to the DistribAI server."""
    script_content = None
    if script_path and Path(script_path).exists():
        script_content = Path(script_path).read_text()

    payload = {
        "org_id": org_id,
        "name": name,
        "job_type": job_type,
        "priority": priority,
        "script_content": script_content,
        "base_model": base_model,
        "dataset_ref": dataset,
        "total_steps": steps,
        "min_gpu_vram_gb": vram_gb,
        "requirements": requirements,
    }

    payload = {k: v for k, v in payload.items() if v is not None}

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    url = f"{server_url}/jobs/submit"
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)


def list_jobs(server_url: str, api_key: str, org_id: str) -> list:
    """List jobs for an organization."""
    headers = {"X-API-Key": api_key}
    url = f"{server_url}/jobs?org_id={org_id}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json().get("jobs", [])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)


def get_job_status(server_url: str, api_key: str, org_id: str, job_id: str) -> dict:
    """Get status of a specific job."""
    headers = {"X-API-Key": api_key}
    url = f"{server_url}/jobs/{job_id}?org_id={org_id}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)


def cancel_job(server_url: str, api_key: str, org_id: str, job_id: str) -> dict:
    """Cancel a job."""
    headers = {"X-API-Key": api_key}
    url = f"{server_url}/jobs/{job_id}/cancel?org_id={org_id}"
    response = requests.post(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="DistribAI Job Submission CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Submit a training job
    python submit_job.py --org my-org --name "GPT Fine-tune" --type finetune \\
        --script train.py --base-model gpt2 --steps 1000

    # List all jobs
    python submit_job.py list --org my-org

    # Check job status
    python submit_job.py status --org my-org --job-id job-abc123

    # Cancel a job
    python submit_job.py cancel --org my-org --job-id job-abc123
        """,
    )

    parser.add_argument(
        "--server",
        default=os.getenv("DISTRIBAI_SERVER_URL", "http://localhost:8766"),
        help="Server URL (default: http://localhost:8766)",
    )
    parser.add_argument(
        "--api-key", default=os.getenv("DISTRIBAI_API_KEY", ""), help="API key for authentication"
    )
    parser.add_argument("--org", required=True, help="Organization ID")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    submit_parser = subparsers.add_parser("submit", help="Submit a new job")
    submit_parser.add_argument("--name", required=True, help="Job name")
    submit_parser.add_argument(
        "--type",
        default="train",
        choices=["train", "finetune", "rl", "inference", "benchmark", "evaluation", "custom"],
        help="Job type",
    )
    submit_parser.add_argument(
        "--priority",
        default="NORMAL",
        choices=["CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND"],
        help="Job priority",
    )
    submit_parser.add_argument("--script", help="Path to script file")
    submit_parser.add_argument("--base-model", help="Base model name")
    submit_parser.add_argument("--dataset", help="Dataset reference")
    submit_parser.add_argument("--steps", type=int, default=1000, help="Total training steps")
    submit_parser.add_argument("--vram", type=float, default=0.0, help="Minimum GPU VRAM (GB)")
    submit_parser.add_argument(
        "--requirement",
        action="append",
        default=[],
        help="Python package requirement (can use multiple times)",
    )

    subparsers.add_parser("list", help="List all jobs")

    status_parser = subparsers.add_parser("status", help="Get job status")
    status_parser.add_argument("--job-id", required=True, help="Job ID")

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("--job-id", required=True, help="Job ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "submit":
        result = submit_job(
            server_url=args.server,
            api_key=args.api_key,
            org_id=args.org,
            name=args.name,
            job_type=args.type,
            priority=args.priority,
            script_path=args.script,
            base_model=args.base_model,
            dataset=args.dataset,
            steps=args.steps,
            vram_gb=args.vram,
            requirements=args.requirement,
        )
        print(f"✓ Job submitted: {result['job_id']}")
        print(f"  Status: {result['status']}")
        if result.get("queue_position"):
            print(f"  Queue position: {result['queue_position']}")

    elif args.command == "list":
        jobs = list_jobs(args.server, args.api_key, args.org)
        print(f"\nJobs for {args.org}:")
        print("-" * 60)
        for job in jobs:
            print(f"  {job['job_id']}: {job['name']}")
            print(f"    Type: {job['type']}, Priority: {job['priority']}, Status: {job['status']}")
            print()

    elif args.command == "status":
        status = get_job_status(args.server, args.api_key, args.org, args.job_id)
        print(f"\nJob: {status['job_id']}")
        print(f"  Name: {status['name']}")
        print(f"  Status: {status['status']}")
        print(f"  Type: {status['job_type']}")
        print(f"  Priority: {status['priority']}")
        print(f"  Progress: {status['progress']:.1f}%")
        if status.get("metrics"):
            print(f"  Metrics: {status['metrics']}")

    elif args.command == "cancel":
        result = cancel_job(args.server, args.api_key, args.org, args.job_id)
        print(f"✓ Job cancelled: {args.job_id}")


if __name__ == "__main__":
    main()
