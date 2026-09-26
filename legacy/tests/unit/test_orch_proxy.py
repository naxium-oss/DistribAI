"""Tests for client/orch-proxy.js admin header and factory behavior."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORCH_PROXY = ROOT / "client" / "orch-proxy.js"


def _run_node(snippet: str) -> dict:
    proc = subprocess.run(
        ["node", "-e", snippet],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout.strip())


def test_orch_admin_headers_use_distribai_secret_only():
    snippet = f"""
    const {{ createOrchProxy }} = require({json.dumps(str(ORCH_PROXY))});
    process.env.DISTRIBAI_ADMIN_SECRET = 'orch-secret';
    process.env.JWT_SECRET = 'jwt-not-admin';
    const {{ orchAdminHeaders }} = createOrchProxy(() => 'http://127.0.0.1:8766');
    console.log(JSON.stringify(orchAdminHeaders()));
    """
    headers = _run_node(snippet)
    assert headers["Authorization"] == "Bearer orch-secret"


def test_orch_admin_headers_omit_authorization_when_secret_unset():
    snippet = f"""
    const {{ createOrchProxy }} = require({json.dumps(str(ORCH_PROXY))});
    delete process.env.DISTRIBAI_ADMIN_SECRET;
    const {{ orchAdminHeaders }} = createOrchProxy(() => 'http://127.0.0.1:8766');
    console.log(JSON.stringify(orchAdminHeaders()));
    """
    headers = _run_node(snippet)
    assert "Authorization" not in headers


def test_fetch_orch_json_parses_admin_nodes():
    snippet = f"""
    const http = require('http');
    const {{ createOrchProxy }} = require({json.dumps(str(ORCH_PROXY))});
    const server = http.createServer((req, res) => {{
      if (req.url === '/admin/nodes') {{
        res.writeHead(200, {{ 'Content-Type': 'application/json' }});
        res.end(JSON.stringify({{ nodes: [{{ node_id: 'n1' }}] }}));
        return;
      }}
      res.writeHead(404);
      res.end();
    }});
    server.listen(0, async () => {{
      const port = server.address().port;
      const {{ fetchOrchJson }} = createOrchProxy(() => `http://127.0.0.1:${{port}}`);
      const body = await fetchOrchJson('/admin/nodes');
      server.close();
      console.log(JSON.stringify(body));
    }});
    """
    body = _run_node(snippet)
    assert body["nodes"][0]["node_id"] == "n1"
