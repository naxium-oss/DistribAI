"""Queue stall diagnostics for admin job listings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from services_python.db_manager import DBManager
from services_python.queue_diagnostics import (
    build_fleet_summary,
    diagnose_job_blockers,
    enrich_jobs_with_queue_hints,
)


def test_no_workers_connected_blocker(tmp_path):
    schema = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "q.db"), str(schema))
    node_service = MagicMock()
    node_service.connected_nodes = {}
    node_service.pending_assignments = {}

    db.create_job(job_type="fine_tune", total_steps=2)
    summary = build_fleet_summary(node_service, db)
    assert summary["queue_depth"] >= 1
    assert summary["connected_count"] == 0

    blockers = diagnose_job_blockers(
        node_service, db, {"status": "queued", "submitter_id": "distribai"}
    )
    codes = {b["code"] for b in blockers}
    assert "no_workers_connected" in codes


def test_idle_worker_scheduler_pending(tmp_path):
    schema = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "q2.db"), str(schema))
    node_service = MagicMock()
    node_service.connected_nodes = {"worker-1"}
    node_service.pending_assignments = {}

    db.create_job(job_type="fine_tune", total_steps=2)
    jobs = db.get_all_jobs()
    enriched, fleet = enrich_jobs_with_queue_hints(node_service, db, jobs)
    assert fleet["idle_count"] == 1
    queued = [j for j in enriched if j.get("status") == "queued"]
    assert queued
    assert any(b["code"] == "scheduler_pending" for b in queued[0].get("queue_blockers", []))
