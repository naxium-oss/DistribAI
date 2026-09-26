from __future__ import annotations

import json
from pathlib import Path

import pytest
from gridlink import providers
from gridlink.providers import (
    LocalProvider,
    TunnelError,
    available_providers,
    cloudflared_command,
    get_provider,
    ngrok_command,
    parse_cloudflared_url,
    parse_ngrok_public_url,
)

CLOUDFLARED_LOG = (
    "2026-09-24T16:00:00Z INF Thank you for trying Cloudflare Tunnel. Doing so, "
    "without a paid account, means you are using it for evaluation purposes.\n"
    "2026-09-24T16:00:01Z INF +---------------------------------------------------------+\n"
    "2026-09-24T16:00:01Z INF |  https://random-words-1234.trycloudflare.com  |\n"
    "2026-09-24T16:00:01Z INF +---------------------------------------------------------+\n"
    "2026-09-24T16:00:01Z INF Registered tunnel connection connIndex=0\n"
)


def test_cloudflared_command_uses_grpc_friendly_flags():
    cmd = cloudflared_command(Path("/usr/local/bin/cloudflared"), "127.0.0.1", 50051)
    assert cmd[0] == "/usr/local/bin/cloudflared"
    assert cmd[1:3] == ["tunnel", "--no-autoupdate"]
    assert "--url" in cmd
    assert cmd[cmd.index("--url") + 1] == "http://127.0.0.1:50051"
    assert "--http2-origin" in cmd


def test_cloudflared_command_can_drop_http2_origin():
    cmd = cloudflared_command(Path("cloudflared"), "127.0.0.1", 50051, http2_origin=False)
    assert "--http2-origin" not in cmd


def test_parse_cloudflared_url_from_real_log():
    assert parse_cloudflared_url(CLOUDFLARED_LOG) == "https://random-words-1234.trycloudflare.com"


def test_parse_cloudflared_url_none_for_other_text():
    assert parse_cloudflared_url("ERROR: something went wrong") is None


def test_ngrok_command_shape():
    cmd = ngrok_command(Path("/usr/bin/ngrok"), "127.0.0.1", 50051)
    assert cmd[0] == "/usr/bin/ngrok"
    assert cmd[1] == "http"
    assert cmd[-1] == "127.0.0.1:50051"


def test_parse_ngrok_public_url_prefers_https():
    payload = {
        "tunnels": [
            {"public_url": "http://abc.ngrok.io", "proto": "http"},
            {"public_url": "https://abc.ngrok-free.app", "proto": "https"},
        ]
    }
    assert parse_ngrok_public_url(payload) == "https://abc.ngrok-free.app"
    assert parse_ngrok_public_url(json.dumps(payload)) == "https://abc.ngrok-free.app"


def test_parse_ngrok_public_url_empty():
    assert parse_ngrok_public_url({"tunnels": []}) is None
    assert parse_ngrok_public_url("not json") is None


def test_local_provider_runs_no_process():
    handle = LocalProvider().start("127.0.0.1", 50051)
    assert handle.url == "127.0.0.1:50051"
    assert handle.process is None
    assert handle.command == []


def test_get_provider_rejects_unknown():
    with pytest.raises(TunnelError):
        get_provider("wireguard")


def test_available_providers_includes_local_and_cloudflare():
    names = {status.name for status in available_providers()}
    assert {"local", "cloudflare", "ngrok"} <= names
    local = next(s for s in available_providers() if s.name == "local")
    assert local.available is True


def test_ngrok_unavailable_without_authtoken(monkeypatch):
    monkeypatch.delenv(providers.NGROK_AUTHTOKEN_ENV, raising=False)
    monkeypatch.delenv("NGROK_BIN", raising=False)
    monkeypatch.setattr(providers, "find_ngrok", lambda: None)
    status = providers.NgrokProvider().available()
    assert status.available is False
    assert "NGROK_AUTHTOKEN" in status.detail


def test_ngrok_missing_token_does_not_break_cloudflare(monkeypatch):
    # The default documented path must never depend on the optional provider.
    monkeypatch.delenv(providers.NGROK_AUTHTOKEN_ENV, raising=False)
    assert providers.PROVIDERS["cloudflare"].available().available is True
