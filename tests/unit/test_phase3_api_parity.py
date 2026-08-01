import asyncio
import json
import socket
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

try:
    from services_python.orchestrator_grpc import serve  # noqa: F401

    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.skipif(not HAS_GRPC, reason="grpc dependencies not available")
@pytest.mark.asyncio
async def test_dashboard_backing_endpoints_exist(monkeypatch):
    from services_python.orchestrator_grpc import serve

    grpc_port = _free_port()
    admin_port = _free_port()
    monkeypatch.setenv("GRPC_PORT", str(grpc_port))
    monkeypatch.setenv("ADMIN_PORT", str(admin_port))
    runtime = await serve()
    base = f"http://127.0.0.1:{admin_port}"
    try:
        docs = await asyncio.to_thread(
            lambda: json.loads(urllib.request.urlopen(f"{base}/api/docs/list", timeout=5).read())
        )
        assert isinstance(docs, list)
        assert any(doc["path"] == "README.md" for doc in docs)

        read_url = f"{base}/api/docs/read?{urllib.parse.urlencode({'path': 'README.md'})}"
        readme = await asyncio.to_thread(
            lambda: json.loads(urllib.request.urlopen(read_url, timeout=5).read())
        )
        assert readme["path"] == "README.md"
        assert readme["content"]

        req = urllib.request.Request(
            f"{base}/api/admin/distribai/registry/sync",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        synced = await asyncio.to_thread(
            lambda: json.loads(urllib.request.urlopen(req, timeout=5).read())
        )
        assert synced["ok"] is True
        assert "synced_count" in synced
        assert "distribai-small" in synced["models"]
    finally:
        await runtime.stop()


def _client_dashboard_sources() -> str:
    root = Path("client")
    parts = [root / "server.js"]
    for subdir in ("routes", "lib"):
        d = root / subdir
        if d.is_dir():
            parts.extend(sorted(d.glob("*.js")))
    return "".join(p.read_text(encoding="utf-8") for p in parts if p.is_file())


def test_desktop_settings_buttons_have_backing_routes():
    server_js = _client_dashboard_sources()
    node_dir = Path("worker/src/dashboard/static/node")
    index_sources = "".join(
        (node_dir / name).read_text(encoding="utf-8")
        for name in ("index.html", "index-preview.js")
        if (node_dir / name).is_file()
    )

    assert "/api/settings/reset-node" in server_js
    assert "/api/settings/unlink-node" in server_js
    assert "fetch('/api/settings/reset-node'" in index_sources
    assert "fetch('/api/settings/unlink-node'" in index_sources


@pytest.mark.skipif(not HAS_GRPC, reason="grpc dependencies not available")
def test_python_sdk_imports_and_uses_live_routes():
    import sys
    from pathlib import Path

    sdk_path = str(Path("sdk/python").resolve())
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)

    import distribai
    from distribai.client import DistribAIClient

    client = DistribAIClient("http://127.0.0.1:8766", api_key="token")

    assert distribai.Client is not None
    assert client.jobs is not None
    assert client.nodes is not None
    assert client.jobs.list.__code__.co_consts
    assert "/admin/jobs" in Path("sdk/python/distribai/jobs.py").read_text(encoding="utf-8")
    assert "/admin/nodes" in Path("sdk/python/distribai/nodes.py").read_text(encoding="utf-8")
