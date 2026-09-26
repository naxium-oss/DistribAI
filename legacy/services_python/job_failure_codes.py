"""Canonical failure and queue-reason codes embedded in admin job JSON."""

from __future__ import annotations

from typing import Any

# Emitted on /admin/jobs via failure_code or queue_blockers[].code
FAILURE_CODES: dict[str, str] = {
    "no_workers_connected": "No gRPC workers are currently connected",
    "workers_offline": "Registered workers are currently offline",
    "all_workers_busy": "Every contributing worker already has work",
    "workers_not_contributing": "Connected workers are not marked contributing",
    "submitter_not_trusted": "Submitter is absent from the trusted list",
    "scheduler_pending": "Idle workers available; scheduler should pick up soon",
    "hash_mismatch": "Script package digest does not match the declared fingerprint",
    "sandbox_install_timeout": "Sandbox dependency install ran past the timeout",
    "sandbox_egress_denied": "Sandbox blocked outbound network access",
    "no_workers_matched_tags": "No worker satisfied the required hardware tags",
    "job_cancelled": "Operator cancelled the job",
    "job_failed": "Job ended in a terminal failure state",
    "task_requeued": "Task returned to queue after a stale assignment",
    "preflight_rejected": "Pre-flight checks rejected the submission",
}


def normalize_failure_code(raw: str | None) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if key in FAILURE_CODES:
        return key
    return key if key else None


def attach_failure_fields(job: dict[str, Any]) -> dict[str, Any]:
    """Copy a job row and attach ``failure_code`` plus the shared catalog hint."""
    row = dict(job)
    reason = (row.get("latest_reason") or "").lower()
    status = (row.get("status") or "").lower()
    code: str | None = None
    if "hash" in reason and "mismatch" in reason:
        code = "hash_mismatch"
    elif "sandbox" in reason and "timeout" in reason:
        code = "sandbox_install_timeout"
    elif "egress" in reason:
        code = "sandbox_egress_denied"
    elif "requeued" in reason:
        code = "task_requeued"
    elif status in ("failed", "error", "timeout"):
        code = "job_failed"
    elif status == "cancelled":
        code = "job_cancelled"
    blockers = row.get("queue_blockers") or []
    if not code and blockers:
        code = blockers[0].get("code")
    if code:
        row["failure_code"] = normalize_failure_code(code)
    row["failure_code_catalog"] = FAILURE_CODES
    return row
