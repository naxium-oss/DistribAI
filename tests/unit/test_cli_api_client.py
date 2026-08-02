"""Unit tests for scripts.cli.api_client.AdminAPIClient."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest import mock

import pytest

from scripts.cli.api_client import AdminAPIClient


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info) -> None:
        return None


@pytest.mark.unit
def test_base_url_defaults_and_strips_trailing_slash(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_ADMIN_URL", raising=False)
    client = AdminAPIClient()
    assert client.base_url == "http://127.0.0.1:8766"

    client = AdminAPIClient("http://example.test:9999/")
    assert client.base_url == "http://example.test:9999"


@pytest.mark.unit
def test_headers_include_bearer_token_when_secret_set(monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ADMIN_SECRET", "s3cr3t")
    client = AdminAPIClient("http://example.test")
    headers = client._headers()
    assert headers["Authorization"] == "Bearer s3cr3t"


@pytest.mark.unit
def test_headers_omit_authorization_when_no_secret(monkeypatch):
    monkeypatch.delenv("DISTRIBAI_ADMIN_SECRET", raising=False)
    client = AdminAPIClient("http://example.test")
    assert "Authorization" not in client._headers()


@pytest.mark.unit
def test_get_returns_decoded_json():
    client = AdminAPIClient("http://example.test")
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({"ok": True})):
        result = client.get("/admin/health")
    assert result == {"ok": True}


@pytest.mark.unit
def test_post_sends_payload_and_returns_json():
    client = AdminAPIClient("http://example.test")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data
        captured["method"] = req.get_method()
        return _FakeResponse({"job_id": "abc"})

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = client.post("/admin/jobs", {"model_name": "m"})

    assert result == {"job_id": "abc"}
    assert captured["method"] == "POST"
    assert json.loads(captured["data"]) == {"model_name": "m"}


@pytest.mark.unit
def test_delete_uses_delete_method():
    client = AdminAPIClient("http://example.test")

    def fake_urlopen(req, timeout=None):
        assert req.get_method() == "DELETE"
        return _FakeResponse({"ok": True})

    with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
        result = client.delete("/admin/jobs/x")
    assert result == {"ok": True}


@pytest.mark.unit
def test_request_empty_body_returns_empty_dict():
    client = AdminAPIClient("http://example.test")

    class _Empty(_FakeResponse):
        def read(self) -> bytes:
            return b""

    with mock.patch("urllib.request.urlopen", return_value=_Empty({})):
        result = client.get("/admin/health")
    assert result == {}


@pytest.mark.unit
def test_http_error_returns_error_dict_from_json_body():
    client = AdminAPIClient("http://example.test")
    body = BytesIO(json.dumps({"error": "Unauthorized org"}).encode("utf-8"))
    http_error = urllib.error.HTTPError("http://example.test", 403, "Forbidden", {}, body)
    with mock.patch("urllib.request.urlopen", side_effect=http_error):
        result = client.get("/admin/jobs")
    assert result == {"error": "Unauthorized org"}


@pytest.mark.unit
def test_http_error_falls_back_to_reason_when_body_not_json():
    client = AdminAPIClient("http://example.test")
    body = BytesIO(b"not json")
    http_error = urllib.error.HTTPError("http://example.test", 500, "Server Error", {}, body)
    with mock.patch("urllib.request.urlopen", side_effect=http_error):
        result = client.get("/admin/jobs")
    assert result == {"error": "HTTP 500: Server Error"}


@pytest.mark.unit
def test_url_error_returns_error_dict():
    client = AdminAPIClient("http://example.test")
    with mock.patch(
        "urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")
    ):
        result = client.get("/admin/health")
    assert "error" in result


@pytest.mark.unit
def test_health_and_is_reachable():
    client = AdminAPIClient("http://example.test")
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({"ok": True})):
        assert client.health() == {"ok": True}
        assert client.is_reachable() is True

    with mock.patch("urllib.request.urlopen", side_effect=OSError("offline")):
        assert client.is_reachable() is False


@pytest.mark.unit
def test_list_nodes_returns_list_or_empty():
    client = AdminAPIClient("http://example.test")
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({"nodes": [{"a": 1}]})):
        assert client.list_nodes() == [{"a": 1}]

    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({"error": "boom"})):
        assert client.list_nodes() == []


@pytest.mark.unit
def test_list_jobs_returns_list_or_empty():
    client = AdminAPIClient("http://example.test")
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({"jobs": [{"job_id": "1"}]})):
        assert client.list_jobs() == [{"job_id": "1"}]

    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({})):
        assert client.list_jobs() == []


@pytest.mark.unit
def test_get_job_delegates_to_job_id_path():
    client = AdminAPIClient("http://example.test")
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({"job_id": "j1"})) as m:
        result = client.get_job("j1")
    assert result == {"job_id": "j1"}
    request_obj = m.call_args[0][0]
    assert request_obj.full_url == "http://example.test/admin/jobs/j1"


@pytest.mark.unit
def test_list_credits_returns_dict_or_empty():
    client = AdminAPIClient("http://example.test")
    with mock.patch(
        "urllib.request.urlopen", return_value=_FakeResponse({"credits": {"n1": {"balance": 1}}})
    ):
        assert client.list_credits() == {"n1": {"balance": 1}}

    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({})):
        assert client.list_credits() == {}


@pytest.mark.unit
def test_tail_logs_returns_list_or_empty():
    client = AdminAPIClient("http://example.test")
    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({"logs": ["a", "b"]})):
        assert client.tail_logs(50) == ["a", "b"]

    with mock.patch("urllib.request.urlopen", return_value=_FakeResponse({})):
        assert client.tail_logs() == []
