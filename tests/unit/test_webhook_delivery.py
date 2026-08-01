"""Unit tests for signed callback_url webhook delivery."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from services_python.webhook_delivery import (
    SIGNATURE_HEADER,
    build_webhook_payload,
    callback_url_allowed,
    deliver_webhook,
    notify_job_terminal,
    schedule_job_callback,
    sign_payload,
    verify_signature,
)


def test_sign_and_verify_roundtrip():
    body = b'{"event":"job.terminal","job_id":"j1"}'
    header = sign_payload(body, signing_key="test-signing-key")
    assert header.startswith("sha256=")
    assert verify_signature(body, header, signing_key="test-signing-key")
    assert not verify_signature(body, "sha256=deadbeef", signing_key="test-signing-key")


def test_build_webhook_payload_from_job_dict():
    payload = build_webhook_payload(
        {"job_id": "job-1", "model_name": "tiny", "priority_tier": "P0"},
        "success",
        "done",
    )
    assert payload["event"] == "job.terminal"
    assert payload["job_id"] == "job-1"
    assert payload["status"] == "success"
    assert payload["reason"] == "done"
    assert payload["source"] == "DistribAI"
    assert payload["priority_tier"] == "P0"


def test_callback_url_rejects_loopback(monkeypatch):
    monkeypatch.delenv("DISTRIBAI_ALLOW_LOOPBACK_WEBHOOKS", raising=False)
    assert callback_url_allowed("https://hooks.example.com/job") is True
    assert callback_url_allowed("http://127.0.0.1/hook") is False
    assert callback_url_allowed("not-a-url") is False


def test_callback_url_allows_loopback_when_enabled(monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ALLOW_LOOPBACK_WEBHOOKS", "1")
    assert callback_url_allowed("http://127.0.0.1:9999/hook") is True


def test_schedule_job_callback_posts_signed_body(monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ALLOW_LOOPBACK_WEBHOOKS", "1")
    monkeypatch.setenv("DISTRIBAI_WEBHOOK_SYNC", "1")
    captured: dict = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["body"] = request.data
        captured["headers"] = dict(request.headers)
        return _Resp()

    job = {
        "job_id": "job-cb",
        "model_name": "m",
        "priority_tier": "P1",
        "hparams": {"callback_url": "http://127.0.0.1:8765/hook"},
    }
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        ok = schedule_job_callback(job, "success", "ok", sync=True, signing_key="k")
    assert ok is True
    assert captured["url"] == "http://127.0.0.1:8765/hook"
    body = captured["body"]
    assert SIGNATURE_HEADER in captured["headers"] or any(
        k.lower() == SIGNATURE_HEADER.lower() for k in captured["headers"]
    )
    # urllib may title-case headers
    sig = captured["headers"].get(SIGNATURE_HEADER) or captured["headers"].get(
        "X-distribai-signature"
    )
    assert sig and verify_signature(body, sig, signing_key="k")
    payload = json.loads(body.decode("utf-8"))
    assert payload["job_id"] == "job-cb"
    assert payload["status"] == "success"


def test_notify_job_terminal_uses_db_hparams(monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ALLOW_LOOPBACK_WEBHOOKS", "1")
    db = MagicMock()
    db.get_job.return_value = {
        "job_id": "job-n",
        "model_name": "n",
        "priority_tier": "P2",
        "progress_pct": 100.0,
    }
    db.get_job_hparams.return_value = {
        "callback_url": "http://127.0.0.1:8765/n",
    }

    with patch(
        "services_python.webhook_delivery.deliver_webhook",
        return_value=(True, 200, "delivered"),
    ) as deliver:
        assert notify_job_terminal(db, "job-n", "failed", "boom", sync=True) is True
        deliver.assert_called_once()
        args, _kwargs = deliver.call_args
        assert args[0] == "http://127.0.0.1:8765/n"
        assert args[1]["status"] == "failed"


def test_notify_skips_without_callback():
    db = MagicMock()
    db.get_job.return_value = {"job_id": "job-x"}
    db.get_job_hparams.return_value = {}
    assert notify_job_terminal(db, "job-x", "success") is False


def test_deliver_rejects_invalid_url():
    ok, code, detail = deliver_webhook("ftp://bad", {"job_id": "j"})
    assert ok is False
    assert code is None
    assert "invalid" in detail


def test_db_get_job_hparams_and_terminal_notify(tmp_path: Path, monkeypatch):
    from services_python.db_manager import DBManager

    monkeypatch.setenv("DISTRIBAI_ALLOW_LOOPBACK_WEBHOOKS", "1")
    schema = Path(__file__).resolve().parents[2] / "runtime" / "db" / "schema.sql"
    db = DBManager(str(tmp_path / "wh.db"), str(schema))
    job_id = db.create_job(
        job_type="fine_tune",
        base_model="distribai-tiny",
        dataset_ref="",
        hyperparams={"callback_url": "http://127.0.0.1:8765/db"},
        total_steps=5,
        model_name="wh",
        priority_tier="P1",
    )
    assert db.get_job_hparams(job_id)["callback_url"].startswith("http://127.0.0.1")

    with patch(
        "services_python.webhook_delivery.schedule_webhook_delivery"
    ) as scheduled:
        db.update_job_status(job_id, "success", "finished")
        assert scheduled.called
