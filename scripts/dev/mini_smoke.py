#!/usr/bin/env python3
"""Mini smoke contract: orchestrator health + admin read paths (release/CI gate).

Run against a live orchestrator (local or remote):

    python -m scripts.dev.mini_smoke
    python -m scripts.dev.mini_smoke --admin-url http://127.0.0.1:8766

When admin auth is enforced, pass the shared secret:

    DISTRIBAI_ADMIN_SECRET=... python -m scripts.dev.mini_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _request(url: str, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _resolve_admin_secret(explicit: str) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    env_secret = os.getenv("DISTRIBAI_ADMIN_SECRET", "").strip()
    if env_secret:
        return env_secret
    return os.getenv("JWT_SECRET", "").strip()


def _admin_headers(secret: str) -> dict[str, str]:
    if not secret:
        return {}
    return {"Authorization": f"Bearer {secret}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DistribAI mini smoke (health + admin reads)")
    parser.add_argument(
        "--admin-url",
        default=os.getenv("ADMIN_URL", "http://127.0.0.1:8766"),
        help="Orchestrator admin base URL (default: http://127.0.0.1:8766)",
    )
    parser.add_argument(
        "--secret",
        default="",
        help="Bearer token when admin auth is enforced (default: DISTRIBAI_ADMIN_SECRET or JWT_SECRET)",
    )
    parser.add_argument(
        "--submit-test-job",
        action="store_true",
        help="POST a minimal job and verify HTTP 200 (requires writable orchestrator)",
    )
    args = parser.parse_args(argv)
    base = args.admin_url.rstrip("/")
    headers = _admin_headers(_resolve_admin_secret(args.secret))

    try:
        health = _request(f"{base}/admin/health", {})
    except urllib.error.URLError as exc:
        print(f"mini_smoke: FAIL — cannot reach {base}/admin/health: {exc}", file=sys.stderr)
        return 1

    if not health.get("ok"):
        print(f"mini_smoke: FAIL — health not ok: {health}", file=sys.stderr)
        return 1

    try:
        nodes = _request(f"{base}/admin/nodes", headers)
        jobs = _request(f"{base}/admin/jobs", headers)
    except urllib.error.HTTPError as exc:
        print(
            f"mini_smoke: FAIL — admin read returned HTTP {exc.code} "
            f"(set DISTRIBAI_ADMIN_SECRET if lockdown is on)",
            file=sys.stderr,
        )
        return 1
    except urllib.error.URLError as exc:
        print(f"mini_smoke: FAIL — admin read error: {exc}", file=sys.stderr)
        return 1

    if args.submit_test_job:
        payload = json.dumps({"steps": 1, "batch_size": 8}).encode()
        post_headers = {**headers, "Content-Type": "application/json"}
        post_req = urllib.request.Request(
            f"{base}/admin/jobs",
            data=payload,
            headers=post_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(post_req, timeout=15) as resp:
                created = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            print(f"mini_smoke: FAIL — job create HTTP {exc.code}", file=sys.stderr)
            return 1
        if not created.get("ok") or not created.get("job_id"):
            print(f"mini_smoke: FAIL — unexpected create response: {created}", file=sys.stderr)
            return 1

    node_count = len(nodes.get("nodes", []))
    job_count = len(jobs.get("jobs", []))
    print(
        f"mini_smoke: ok — health=True nodes={node_count} jobs={job_count} "
        f"queued={health.get('queued_jobs', '?')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
