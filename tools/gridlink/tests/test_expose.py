from __future__ import annotations

import gridlink.expose as expose_mod
from gridlink.expose import ExposeSession, expose, is_loopback
from gridlink.providers import TunnelHandle


class _StubProvider:
    name = "cloudflare"

    def __init__(self, url: str) -> None:
        self._url = url
        self.started = False

    def start(self, bind_host, port, *, timeout=45.0, log=print):  # noqa: ANN001
        self.started = True
        return TunnelHandle(
            provider=self.name,
            url=self._url,
            command=["cloudflared", "tunnel", "--url", f"http://{bind_host}:{port}"],
            process=None,
        )


def test_local_is_the_default_and_exposes_nothing():
    result = ExposeSession(provider="local").start(log=lambda *_: None)
    assert result.provider == "local"
    assert result.public_url is None
    assert result.link.host == "127.0.0.1"
    assert result.link.tls is False
    assert result.command == []
    result.stop()  # no-op


def test_local_link_host_override():
    result = expose(provider="local", link_host="192.168.1.20", port=60000, log=lambda *_: None)
    assert result.link.orchestrator_url == "http://192.168.1.20:60000"
    assert result.link.engine == "native"


def test_legacy_engine_keeps_the_grpc_target():
    result = expose(
        provider="local", port=50051, engine="legacy", link_host="192.168.1.20", log=lambda *_: None
    )
    assert result.link.orchestrator_url == "192.168.1.20:50051"
    assert result.link.engine == "legacy"


def test_cloudflare_tunnel_becomes_a_tls_link(monkeypatch):
    stub = _StubProvider("https://random-words-1234.trycloudflare.com")
    monkeypatch.setattr(expose_mod, "get_provider", lambda name: stub)

    session = ExposeSession(provider="cloudflare", name="my-grid", invite="team")
    result = session.start(log=lambda *_: None)

    assert stub.started is True
    assert result.provider == "cloudflare"
    assert result.public_url == "https://random-words-1234.trycloudflare.com"
    assert result.link.host == "random-words-1234.trycloudflare.com"
    assert result.link.port == 443
    assert result.link.tls is True
    assert result.link.invite == "team"
    assert result.link.is_public
    assert "cloudflared" in result.command[0]


def test_stop_terminates_only_when_a_tunnel_exists(monkeypatch):
    stub = _StubProvider("https://x.trycloudflare.com")
    monkeypatch.setattr(expose_mod, "get_provider", lambda name: stub)
    session = ExposeSession(provider="cloudflare")
    session.start(log=lambda *_: None)
    session.stop()  # handle has process=None -> safe
    # local sessions have no handle at all
    ExposeSession(provider="local").stop()


def test_expose_json_shape(monkeypatch):
    stub = _StubProvider("https://abc.trycloudflare.com")
    monkeypatch.setattr(expose_mod, "get_provider", lambda name: stub)
    payload = expose(provider="cloudflare", log=lambda *_: None).to_dict()
    assert payload["provider"] == "cloudflare"
    assert payload["engine"] == "native"
    assert payload["tls"] is True
    assert payload["orchestrator_url"] == "https://abc.trycloudflare.com:443"
    assert payload["link"].startswith("distribai://join?")


def test_is_loopback():
    assert is_loopback("127.0.0.1")
    assert is_loopback("localhost")
    assert not is_loopback("10.0.0.5")
