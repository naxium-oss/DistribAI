"""Unit tests for scheduler priority lanes P0–P3."""

from __future__ import annotations

from pathlib import Path

from services_python.db_manager import DBManager
from services_python.priority_lanes import (
    normalize_priority_tier,
    parse_priority_tier_filter,
    priority_lane_rank,
)


def _db(tmp_path: Path) -> DBManager:
    schema = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    return DBManager(str(tmp_path / "lanes.db"), str(schema))


def test_priority_lane_rank_order():
    assert priority_lane_rank("P0") < priority_lane_rank("P1")
    assert priority_lane_rank("P1") < priority_lane_rank("P2")
    assert priority_lane_rank("P2") < priority_lane_rank("P3")
    assert normalize_priority_tier("p0") == "P0"
    assert normalize_priority_tier("2") == "P2"


def test_parse_priority_tier_filter_csv():
    assert parse_priority_tier_filter(None) is None
    assert parse_priority_tier_filter("all") is None
    assert parse_priority_tier_filter("P0,P2") == {"P0", "P2"}


def test_get_queued_tasks_orders_all_lanes(tmp_path: Path):
    db = _db(tmp_path)
    order = ["P3", "P1", "P2", "P0"]
    for tier in order:
        db.create_job(
            job_type="fine_tune",
            base_model="distribai-tiny",
            dataset_ref="",
            hyperparams={},
            total_steps=4,
            model_name=f"job-{tier}",
            priority_tier=tier,
            priority=1,
        )
    tasks = db.get_queued_tasks()
    tiers = [t["priority_tier"] for t in tasks]
    assert tiers == ["P0", "P1", "P2", "P3"]


def test_public_queue_prefers_higher_lanes(tmp_path: Path):
    db = _db(tmp_path)
    db.create_job(
        job_type="fine_tune",
        base_model="distribai-tiny",
        dataset_ref="",
        hyperparams={},
        total_steps=2,
        model_name="low",
        priority_tier="P3",
        priority=50,
    )
    db.create_job(
        job_type="fine_tune",
        base_model="distribai-tiny",
        dataset_ref="",
        hyperparams={},
        total_steps=2,
        model_name="high",
        priority_tier="P0",
        priority=1,
    )
    queue = db.get_public_queue()
    active = [j for j in queue if j["status"] in {"queued", "assigned", "running"}]
    assert active[0]["priority"] == "P0"
