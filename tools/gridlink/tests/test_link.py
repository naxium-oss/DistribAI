from __future__ import annotations

import pytest
from gridlink.link import DEFAULT_GRID_PORT, DEFAULT_GRPC_PORT, GridLink, default_port


def test_cloudflare_link_round_trips():
    link = GridLink(
        host="random-words.trycloudflare.com",
        port=443,
        tls=True,
        provider="cloudflare",
        public_url="https://random-words.trycloudflare.com",
        invite="team-alpha",
        name="my-grid",
    )
    parsed = GridLink.parse(link.to_uri())
    assert parsed == link
    assert parsed.engine == "native"
    assert parsed.orchestrator_url == "https://random-words.trycloudflare.com:443"
    assert parsed.is_public


def test_legacy_link_round_trips():
    link = GridLink(host="grid.example.com", port=DEFAULT_GRPC_PORT, engine="legacy", tls=True)
    parsed = GridLink.parse(link.to_uri())
    assert parsed == link
    assert parsed.orchestrator_url == "grid.example.com:50051"
    assert parsed.is_native is False


def test_default_ports_per_engine():
    assert default_port("native") == DEFAULT_GRID_PORT
    assert default_port("legacy") == DEFAULT_GRPC_PORT


def test_bare_host_port_is_local_and_native():
    link = GridLink.parse("10.0.0.5:50051")
    assert link.host == "10.0.0.5"
    assert link.port == 50051
    assert link.tls is False
    assert link.provider == "local"
    assert link.engine == "native"


def test_bare_host_uses_the_native_default_port():
    link = GridLink.parse("my-laptop.local")
    assert link.host == "my-laptop.local"
    assert link.port == DEFAULT_GRID_PORT


def test_http_url_is_native_and_plaintext():
    link = GridLink.parse("http://127.0.0.1:50061")
    assert link.engine == "native"
    assert link.tls is False
    assert link.port == 50061
    assert link.orchestrator_url == "http://127.0.0.1:50061"


def test_https_url_is_tls_and_provider_tunnel():
    link = GridLink.parse("https://abc123.ngrok-free.app")
    assert link.host == "abc123.ngrok-free.app"
    assert link.port == 443
    assert link.tls is True
    assert link.provider == "tunnel"
    assert link.engine == "native"


def test_grpc_url_selects_the_legacy_engine():
    link = GridLink.parse("grpc://grid.example.com")
    assert link.engine == "legacy"
    assert link.port == DEFAULT_GRPC_PORT
    assert link.orchestrator_url == "grid.example.com:50051"


def test_uri_tls_flag_is_respected():
    uri = "distribai://join?v=1&provider=cloudflare&host=abc.trycloudflare.com&port=443&tls=0"
    link = GridLink.parse(uri)
    assert link.tls is False
    assert link.host == "abc.trycloudflare.com"


def test_uri_without_tls_defaults_for_public_providers():
    uri = "distribai://join?v=1&provider=cloudflare&host=abc.trycloudflare.com&port=443"
    assert GridLink.parse(uri).tls is True


def test_uri_without_engine_defaults_to_native():
    uri = "distribai://join?v=1&host=127.0.0.1&port=50061"
    assert GridLink.parse(uri).engine == "native"


def test_unknown_engine_is_rejected():
    uri = "distribai://join?v=1&engine=cobol&host=127.0.0.1&port=50061"
    with pytest.raises(ValueError, match="unknown engine"):
        GridLink.parse(uri)


def test_whitespace_and_quotes_are_tolerated():
    assert GridLink.parse("  'host.example:9000'  ").orchestrator_url == "http://host.example:9000"


@pytest.mark.parametrize("bad", ["", "   ", "distribai://join?port=443"])
def test_invalid_links_raise(bad):
    with pytest.raises(ValueError):
        GridLink.parse(bad)


def test_share_block_shows_the_native_worker_command():
    link = GridLink(
        host="x.trycloudflare.com",
        port=443,
        tls=True,
        provider="cloudflare",
        invite="code",
    )
    block = link.share_block()
    assert link.to_uri() in block
    assert "distribai_worker --orchestrator https://x.trycloudflare.com:443" in block
    assert "--invite code" in block
    assert "make grid" in block
    assert "free" in block.lower()


def test_legacy_share_block_shows_the_env_vars():
    link = GridLink(
        host="grid.example.com",
        port=DEFAULT_GRPC_PORT,
        engine="legacy",
        tls=True,
        provider="cloudflare",
        invite="code",
    )
    block = link.share_block()
    assert "ORCHESTRATOR_URL=grid.example.com:50051" in block
    assert "GRPC_USE_TLS=true" in block
    assert "DISTRIBAI_INVITE_CODE=code" in block
