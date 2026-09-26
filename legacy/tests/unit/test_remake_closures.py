"""Unit tests for webhook signing, credit estimates, notebook mount, priority lanes."""

from __future__ import annotations

import json
from pathlib import Path

from services_python.credits_estimator import estimate_job_credits
from services_python.webhook_delivery import (
    build_webhook_payload,
    sign_payload,
    validate_callback_url,
)
from worker.src.sandbox.notebook_mount import ipynb_to_python


def test_credit_estimate_scales_with_steps_and_tier():
    low = estimate_job_credits({"steps": 100, "batch_size": 8, "priority_tier": "P3"})
    high = estimate_job_credits({"steps": 1000, "batch_size": 8, "priority_tier": "P0"})
    assert low["estimate"] is True
    assert high["credits"] > low["credits"]
    assert high["priority_tier"] == "P0"


def test_webhook_signature_is_stable():
    body = b'{"job_id":"j1","status":"success"}'
    assert sign_payload(body).startswith("sha256=")
    assert sign_payload(body) == sign_payload(body)


def test_webhook_payload_shape():
    payload = build_webhook_payload(
        {"job_id": "abc", "priority_tier": "P1"},
        "success",
        "ok",
    )
    assert payload["event"] == "job.terminal"
    assert payload["job_id"] == "abc"
    assert payload["status"] == "success"


def test_callback_url_validation():
    assert validate_callback_url("https://hooks.example.com/job") == "https://hooks.example.com/job"
    assert validate_callback_url("ftp://x") is None
    assert validate_callback_url("") is None


def test_notebook_extracts_code_cells(tmp_path: Path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# hi"]},
            {"cell_type": "code", "source": ["x = 1\n", "print(x)\n"]},
            {"cell_type": "code", "source": ["y = x + 1\n"]},
        ]
    }
    path = tmp_path / "job.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    source = ipynb_to_python(path.read_text(encoding="utf-8"))
    assert "x = 1" in source
    assert "y = x + 1" in source


def test_priority_lane_sql_order_prefers_p0(tmp_path: Path):
    from services_python.db_manager import DBManager

    schema = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "lanes.db"), str(schema))
    db.create_job(
        job_type="fine_tune",
        base_model="distribai-tiny",
        dataset_ref="",
        hyperparams={},
        total_steps=10,
        model_name="low",
        priority_tier="P3",
        priority=1,
    )
    db.create_job(
        job_type="fine_tune",
        base_model="distribai-tiny",
        dataset_ref="",
        hyperparams={},
        total_steps=10,
        model_name="high",
        priority_tier="P0",
        priority=1,
    )
    tasks = db.get_queued_tasks()
    assert tasks, "expected queued tasks"
    assert tasks[0]["priority_tier"] == "P0"


def test_weight_export_torch(tmp_path: Path):
    import torch.nn as nn

    from worker.src.compute.weight_export import export_state_dict_torch

    model = nn.Linear(4, 2)
    out = export_state_dict_torch(model, tmp_path / "m.pt")
    assert out.is_file()
