"""
inject_job.py — Manually inject training jobs into the orchestrator.
Usage:
  python tools/inject_job.py
  python tools/inject_job.py --count 5
  python tools/inject_job.py --model distribai-large --steps 100 --preset standard
  python tools/inject_job.py --admin http://192.168.1.10:8766 --count 10
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def inject(admin_url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{admin_url}/admin/jobs",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection failed: {e.reason}", file=sys.stderr)
        print(
            "Is the orchestrator running?  python -m services_python.orchestrator_grpc",
            file=sys.stderr,
        )
        sys.exit(1)


def list_jobs(admin_url: str) -> None:
    with urllib.request.urlopen(f"{admin_url}/admin/jobs", timeout=5) as resp:
        data = json.loads(resp.read())
    jobs = data.get("jobs", [])
    if not jobs:
        print("No jobs yet.")
        return
    print(f"\n{'JOB ID':<20} {'STATUS':<12} {'MODEL':<22} {'STEPS':>6} {'ASSIGNED TO'}")
    print("─" * 78)
    for j in jobs[:20]:
        print(
            f"{j['job_id']:<20} {j['status']:<12} {j['model_name']:<22} "
            f"{j['steps']:>6}  {j.get('assigned_to') or '—'}"
        )
    if len(jobs) > 20:
        print(f"  … {len(jobs) - 20} more")


def list_nodes(admin_url: str) -> None:
    with urllib.request.urlopen(f"{admin_url}/admin/nodes", timeout=5) as resp:
        data = json.loads(resp.read())
    nodes = data.get("nodes", [])
    if not nodes:
        print("No nodes connected.")
        return
    print(f"\n{'NODE ID':<24} {'STATUS':<12} {'GPU':<28} {'DONE':>5} {'FAIL':>5}")
    print("─" * 78)
    for n in nodes:
        print(
            f"{n['node_id']:<24} {n['status']:<12} "
            f"{n['hardware'].get('gpu_model', '?'):<28} "
            f"{n['jobs_completed']:>5} {n['jobs_failed']:>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject training jobs into orchestrator")
    parser.add_argument("--admin", default="http://localhost:8766", help="Admin API URL")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--preset",
        default="quick",
        choices=["quick", "standard", "large"],
        help="Job preset (default: quick)",
    )
    parser.add_argument("--deadline", type=int, default=None, help="Deadline in seconds")
    parser.add_argument("--count", "-n", type=int, default=1, help="Number of jobs to inject")
    parser.add_argument("--list-jobs", action="store_true", help="List current jobs")
    parser.add_argument("--list-nodes", action="store_true", help="List connected nodes")
    args = parser.parse_args()
    if args.list_jobs:
        list_jobs(args.admin)
        return
    if args.list_nodes:
        list_nodes(args.admin)
        return
    payload: dict = {"preset": args.preset}
    if args.model:
        payload["model_name"] = args.model
    if args.steps is not None:
        payload["steps"] = args.steps
    if args.batch_size is not None:
        payload["batch_size"] = args.batch_size
    if args.deadline is not None:
        payload["deadline_s"] = args.deadline
    print(f"Injecting {args.count} job(s) → {args.admin}")
    print(f"Preset: {args.preset}  payload: {payload}\n")
    for i in range(1, args.count + 1):
        result = inject(args.admin, dict(payload))
        print(
            f"[{i}/{args.count}] job_id={result.get('job_id')} "
            f"model={result.get('model_name')} steps={result.get('steps')} "
            f"queue_depth={result.get('queue_depth')}"
        )
    print("\nDone. Check status: python tools/inject_job.py --list-jobs")
    print(f"Live stream:        curl -N {args.admin}/admin/stream")


if __name__ == "__main__":
    main()
