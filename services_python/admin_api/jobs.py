"""Training-job queue: create, list, cancel, retry, artifacts, cleanup."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from aiohttp import web

from services_python.bundle_store import bundle_root, save_bundle
from services_python.constants import DEFAULT_STEPS_PER_TASK
from services_python.credits_estimator import estimate_job_credits
from services_python.db_manager import DBManager
from services_python.job_failure_codes import attach_failure_fields
from services_python.pagination import PaginationHeaders, paginate_list, parse_pagination_params
from services_python.preflight import validate_script_tarball
from services_python.priority_lanes import parse_priority_tier_filter
from services_python.queue_diagnostics import diagnose_job_blockers, enrich_jobs_with_queue_hints
from services_python.schemas import validate_job_create

if TYPE_CHECKING:
    from services_python.orchestrator_grpc import NodeService

logger = logging.getLogger(__name__)


class JobsHandler:
    """Operator job lifecycle against SQLite plus queue diagnostics."""

    def __init__(self, db: DBManager, node_service: NodeService) -> None:
        self.db = db
        self.node_service = node_service

    async def list(self, req: web.Request) -> web.Response:
        """Filtered job list with optional history and paging window."""
        self.node_service._authenticate_request(req, required_kind="admin")
        jobs = await asyncio.to_thread(self.db.get_all_jobs)
        include_history = req.query.get("include_history", "false").lower() == "true"
        active_only = req.query.get("active_only", "true").lower() != "false"
        status_filter = (req.query.get("status") or "").strip()
        model_filter = (req.query.get("model") or "").strip().lower()
        tier_wanted = parse_priority_tier_filter(
            req.query.get("priority_tier") or req.query.get("lane")
        )
        sort_order = (req.query.get("sort") or "newest").strip().lower()

        if status_filter and status_filter.lower() not in {"all", "*"}:
            wanted = {s.strip().lower() for s in status_filter.split(",") if s.strip()}
            if "completed" in wanted:
                wanted.discard("completed")
                wanted.update({"success"})
            jobs = [job for job in jobs if str(job.get("status", "")).lower() in wanted]
        elif status_filter.lower() in {"all", "*"}:
            pass  # keep full history for explicit "all"
        elif active_only and not include_history:
            terminal = {"cancelled", "success", "failed", "timeout"}
            jobs = [job for job in jobs if job.get("status") not in terminal]

        if model_filter:
            jobs = [
                job
                for job in jobs
                if model_filter in str(job.get("model_name") or "").lower()
            ]
        if tier_wanted:
            jobs = [
                job
                for job in jobs
                if str(job.get("priority_tier") or "").upper() in tier_wanted
            ]

        reverse = sort_order in {"newest", "desc", "-created_ts"}
        jobs = sorted(jobs, key=lambda j: int(j.get("created_ts") or 0), reverse=reverse)

        if not include_history:
            for job in jobs:
                job.pop("loss_history", None)

        jobs, fleet = await asyncio.to_thread(
            enrich_jobs_with_queue_hints, self.node_service, self.db, jobs
        )
        jobs = [attach_failure_fields(j) for j in jobs]

        try:
            page = max(1, int(req.query.get("page", "1")))
        except ValueError:
            page = 1
        try:
            per_page = min(100, max(1, int(req.query.get("per_page", "25"))))
        except ValueError:
            per_page = 25
        total = len(jobs)
        total_pages = max(1, (total + per_page - 1) // per_page)
        start = (page - 1) * per_page
        page_jobs = jobs[start : start + per_page]

        return web.json_response(
            {
                "jobs": page_jobs,
                "queue_depth": fleet["queue_depth"],
                "queue_fleet": fleet,
                "total_jobs": total,
                "total_pages": total_pages,
                "page": page,
                "per_page": per_page,
                "depth": fleet["queue_depth"],
            }
        )

    async def get(self, req: web.Request) -> web.Response:
        """Single job row looked up by path job_id."""
        self.node_service._authenticate_request(req, required_kind="admin")
        job_id = req.match_info.get("job_id")
        if not job_id:
            return web.json_response({"error": "missing job_id"}, status=400)

        job = await asyncio.to_thread(self.db.get_job, job_id)
        if not job:
            return web.json_response({"error": "not found"}, status=404)

        blockers = await asyncio.to_thread(
            diagnose_job_blockers, self.node_service, self.db, job
        )
        if blockers:
            job = dict(job)
            job["queue_blockers"] = blockers
        return web.json_response(attach_failure_fields(job))

    async def create(self, req: web.Request) -> web.Response:
        """Schema-check body and insert a new queued training job."""
        self.node_service._authenticate_request(req, required_kind="admin")
        try:
            body = await req.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid JSON"}, status=400)

        if not isinstance(body, dict):
            return web.json_response({"error": "request body must be an object"}, status=400)
        body = dict(body)
        raw_hparams = body.get("hparams")
        if raw_hparams is None:
            raw_hparams = body.get("hyperparams")
        if raw_hparams is not None and not isinstance(raw_hparams, dict):
            return web.json_response({"error": "hparams must be an object"}, status=400)
        hparams = dict(raw_hparams or {})
        # Legacy payloads use ``hyperparams``; normalize to ``hparams`` before
        # schema validation so persistence always stores one field name.
        body["hparams"] = hparams
        body.pop("hyperparams", None)

        script_pkg: bytes | None = None
        script_b64 = body.get("script_package_b64")
        if script_b64:
            try:
                script_pkg = base64.b64decode(script_b64, validate=True)
            except (ValueError, binascii.Error):
                return web.json_response({"error": "invalid script_package_b64"}, status=400)
            if not script_pkg:
                return web.json_response({"error": "empty script package"}, status=400)
            ok, preflight_err, _meta = validate_script_tarball(script_pkg)
            if not ok:
                return web.json_response(
                    {"error": preflight_err, "failure_code": "preflight_rejected"},
                    status=400,
                )
            hparams["execution_paradigm"] = "script"
            body["hparams"] = hparams

        script_content = body.get("script_content")
        if isinstance(script_content, str) and script_content.strip():
            from services_python.script_validation import validate_submitted_script

            err_codes, hints = validate_submitted_script(script_content)
            if err_codes:
                return web.json_response(
                    {
                        "error": "script validation failed",
                        "failure_code": "script_validation_rejected",
                        "validation_errors": err_codes,
                        "suggestions": hints,
                    },
                    status=400,
                )
            hparams["execution_paradigm"] = "script"
            body["hparams"] = hparams

        valid, error, validated = validate_job_create(body)
        if not valid:
            return web.json_response({"error": error}, status=400)
        if hasattr(validated, "model_dump"):
            job_req = validated.model_dump()
        else:
            job_req = vars(validated)
        architecture_config = job_req.get("architecture_config")
        merged_hparams = dict(job_req.get("hparams") or hparams)
        if architecture_config is not None:
            merged_hparams["architecture_config"] = architecture_config

        job_extra = {
            key: job_req[key]
            for key in (
                "model_name",
                "description",
                "batch_size",
                "priority",
                "priority_tier",
                "submitter_id",
                "org",
                "deadline_seconds",
                "max_attempts",
                "steps_per_task",
                "batch_blob_url",
                "weight_blob_url",
            )
            if job_req.get(key) is not None
        }

        job_id = await asyncio.to_thread(
            self.db.create_job,
            job_type=job_req.get("job_type", "fine_tune"),
            base_model=job_req.get("base_model", ""),
            dataset_ref=job_req.get("dataset_ref", ""),
            hyperparams=merged_hparams,
            total_steps=job_req.get("steps", DEFAULT_STEPS_PER_TASK),
            **job_extra,
        )

        job_row = await asyncio.to_thread(self.db.get_job, job_id)
        task_id = (job_row or {}).get("latest_task_id")
        task_ids = [
            task["task_id"]
            for task in (job_row or {}).get("tasks", [])
            if isinstance(task.get("task_id"), str) and task.get("task_id")
        ]
        if script_pkg and task_ids:
            try:
                for current_task_id in task_ids:
                    await asyncio.to_thread(save_bundle, current_task_id, script_pkg)
            except (OSError, ValueError) as exc:
                logger.exception("Failed to persist script bundle for %s: %s", job_id, exc)
                return web.json_response(
                    {"error": "failed to persist script bundle", "job_id": job_id},
                    status=500,
                )
            for current_task_id in task_ids:
                self.node_service.script_packages[current_task_id] = script_pkg
        payload: dict[str, object] = {"ok": True, "job_id": job_id}
        if isinstance(task_id, str) and task_id:
            payload["task_id"] = task_id
        return web.json_response(payload)

    async def cancel(self, req: web.Request) -> web.Response:
        """Cancel a job and halt additional task assignment."""
        self.node_service._authenticate_request(req, required_kind="admin")
        job_id = req.match_info.get("job_id")
        if not job_id:
            return web.json_response({"error": "missing job_id"}, status=400)

        cancelled = await asyncio.to_thread(self.db.cancel_job, job_id)
        if not cancelled:
            return web.json_response({"error": "not found", "cancelled": False}, status=404)
        return web.json_response({"ok": True, "cancelled": True, "job_id": job_id})

    async def recalculate_priorities(self, req: web.Request) -> web.Response:
        """Recompute queue ordering from votes and tier weights."""
        self.node_service._authenticate_request(req, required_kind="admin")
        await asyncio.to_thread(self.db.refresh_queue_positions)
        depth = await asyncio.to_thread(self.db.get_queue_depth)
        return web.json_response({"ok": True, "queue_depth": depth})

    async def clear_completed(self, req: web.Request) -> web.Response:
        """Purge terminal jobs (done/failed/cancelled/timeout) from operator storage."""
        self.node_service._authenticate_request(req, required_kind="admin")
        removed = await asyncio.to_thread(self.db.clear_completed_jobs)
        return web.json_response({"ok": True, "removed": removed})

    async def list_paginated(self, req: web.Request) -> web.Response:
        """Page through jobs when the queue is too large for a single response."""
        self.node_service._authenticate_request(req, required_kind="admin")
        allowed = {"job_id", "status", "created_ts", "updated_ts"}
        params = parse_pagination_params(dict(req.query), allowed_sort_columns=allowed)

        jobs = await asyncio.to_thread(self.db.get_all_jobs)
        result = paginate_list(jobs, params)

        headers = PaginationHeaders.build(
            total=result.pagination["total"], page=params.page, per_page=params.per_page
        )

        return web.json_response(result.to_dict(), headers=headers)

    async def compare(self, req: web.Request) -> web.Response:
        """Side-by-side job compare using query params a and b."""
        self.node_service._authenticate_request(req, required_kind="admin")
        job_a = req.query.get("a", "").strip()
        job_b = req.query.get("b", "").strip()
        if not job_a or not job_b:
            return web.json_response({"error": "query params a and b required"}, status=400)

        row_a = await asyncio.to_thread(self.db.get_job, job_a)
        row_b = await asyncio.to_thread(self.db.get_job, job_b)
        if not row_a or not row_b:
            return web.json_response({"error": "one or both jobs not found"}, status=404)

        def _summary(row: dict) -> dict:
            return {
                "job_id": row.get("job_id"),
                "status": row.get("status"),
                "model_name": row.get("model_name"),
                "steps": row.get("steps"),
                "total_steps": row.get("total_steps"),
                "progress_pct": row.get("progress_pct"),
                "created_ts": row.get("created_ts"),
                "completed_ts": row.get("completed_ts"),
                "latest_reason": row.get("latest_reason"),
                "failure_code": attach_failure_fields(row).get("failure_code"),
            }

        return web.json_response({"a": _summary(row_a), "b": _summary(row_b)})

    async def retry(self, req: web.Request) -> web.Response:
        """Re-queue a terminal/failed job for another assignment pass."""
        self.node_service._authenticate_request(req, required_kind="admin")
        job_id = req.match_info.get("job_id")
        if not job_id:
            return web.json_response({"error": "missing job_id"}, status=400)

        job = await asyncio.to_thread(self.db.get_job, job_id)
        if not job:
            return web.json_response({"error": "not found"}, status=404)

        terminal = {"cancelled", "success", "failed", "timeout", "error"}
        if job.get("status") not in terminal:
            return web.json_response(
                {"error": "job is not in a terminal state", "status": job.get("status")},
                status=409,
            )

        result = await asyncio.to_thread(self.db.operator_retry_job, job_id)
        if result is None:
            return web.json_response({"error": "not found"}, status=404)
        if not result.get("requeued"):
            return web.json_response(
                {
                    "error": "no terminal tasks to re-queue",
                    "job_id": job_id,
                    "status": job.get("status"),
                },
                status=409,
            )
        return web.json_response(
            {
                "ok": True,
                "job_id": job_id,
                "status": "queued",
                "requeued_tasks": result["requeued"],
            }
        )

    async def artifacts(self, req: web.Request) -> web.Response:
        """Artifact and checkpoint paths for the job's newest task."""
        self.node_service._authenticate_request(req, required_kind="admin")
        job_id = req.match_info.get("job_id")
        if not job_id:
            return web.json_response({"error": "missing job_id"}, status=400)

        job = await asyncio.to_thread(self.db.get_job, job_id)
        if not job:
            return web.json_response({"error": "not found"}, status=404)

        task_id = job.get("latest_task_id")
        artifacts: list[dict[str, str]] = []
        if isinstance(task_id, str) and task_id:
            bundle_path = bundle_root() / f"{task_id}.tar.gz"
            if bundle_path.is_file():
                artifacts.append(
                    {"kind": "script_bundle", "path": str(bundle_path), "task_id": task_id}
                )
            ckpt = (
                Path(__file__).resolve().parents[2]
                / "runtime"
                / "checkpoints"
                / f"task_{task_id}_grads.json"
            )
            if ckpt.is_file():
                artifacts.append({"kind": "gradients", "path": str(ckpt), "task_id": task_id})

        return web.json_response({"job_id": job_id, "task_id": task_id, "artifacts": artifacts})

    async def estimate_cost(self, req: web.Request) -> web.Response:
        """Return a non-authoritative credit estimate for the create-job UI."""
        self.node_service._authenticate_request(req, required_kind="admin")
        try:
            body = await req.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return web.json_response(estimate_job_credits(body))
