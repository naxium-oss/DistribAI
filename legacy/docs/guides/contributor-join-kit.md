# Contributor join kit

> **Deprecated.** This page documents the Python-era stack under `legacy/`.
> It receives bug fixes only. For current documentation start at
> [`docs/README.md`](../../../docs/README.md), and for the mapping from a
> Python module to its C++ replacement see
> [`docs/from-python-to-cpp.md`](../../../docs/from-python-to-cpp.md).

How contributors join a **private orchestrator** without ever receiving backend source code.

> **Faster free path:** [`tools/gridlink`](../../../tools/gridlink/README.md) does
> this in one command: `gridlink expose --provider cloudflare` publishes the
> gRPC control plane through a free Cloudflare quick tunnel (no account, no
> custom domain) and prints a shareable `distribai://join?...` link that
> contributors paste into `gridlink join` or a notebook template. Local bind
> stays the default. The manual env-var kit below still applies.

## Who runs what

| Role | Runs | Gets source? |
|------|------|--------------|
| Grid operator ("boss") | `services_python.orchestrator_grpc` + admin HTTP | Private repo / admin binary only |
| Contributors | Worker daemon (`worker.src.daemon.run` or `distribai-node` binary) | Public worker tree or release binary |
| Job submitters (optional) | Python SDK / CLI against admin API | `sdk/python` only |

The public mirror ([`scripts/publish/publish_public_grid.py`](../../scripts/publish/publish_public_grid.py)) ships `worker/`, `client/`, `sdk/python`, and docs, but **not** [`services_python/`](../../services_python/).

## What the operator publishes

Give contributors a **join kit** (paste into email, wiki, or notebook README):

1. **gRPC target**: `ORCHESTRATOR_URL=your-host.example.com:50051`
2. **Admin URL**: `ADMIN_URL=https://your-host.example.com` (or `http://host:8766` on a private LAN)
3. **TLS trust**: `GRPC_USE_TLS=true` and `GRPC_TLS_CA=/path/to/orchestrator-ca.pem` (or embed CA in notebook)
4. **Invite code** (recommended): `DISTRIBAI_INVITE_CODE=team-alpha-2026`
5. **Worker install**, one of:
   - Download **node binary** from your GitHub Release ([update-hosting.md](update-hosting.md))
   - Clone **public grid repo** + `pip install -r requirements-worker.txt`
   - Open [Colab/Kaggle template](../../examples/colab/join_grid.ipynb) ([ephemeral guide](ephemeral-compute-colab-kaggle.md))

Do **not** share orchestrator `.env`, `DISTRIBAI_ADMIN_SECRET`, or `services_python/` tree.

## Worker environment variables

These are read by [`worker/src/daemon/run.py`](../../worker/src/daemon/run.py) and [`worker/src/daemon/daemon.py`](../../worker/src/daemon/daemon.py):

| Variable | Required | Purpose |
|----------|----------|---------|
| `ORCHESTRATOR_URL` | yes | gRPC host:port (default `localhost:50051`) |
| `ADMIN_URL` | recommended | Full admin base URL for registration (e.g. `http://host:8766`) |
| `DISTRIBAI_API_URL` | optional | Override admin URL if different from derived default |
| `DISTRIBAI_JWT_TOKEN` | optional | Skip registration when pre-issued |
| `DISTRIBAI_INVITE_CODE` | optional | Invite gate on registration |
| `DISTRIBAI_EPHEMERAL` | optional | `1` for Colab/Kaggle: temp state, in-memory auth ([ephemeral guide](ephemeral-compute-colab-kaggle.md)) |
| `STATE_DIR` | optional | Persistent worker state (CLI: `--state-dir`) |
| `NODE_ID` | optional | Fixed node id (CLI: `--node-id`) |
| `GRPC_USE_TLS` | public grids | Must match orchestrator |
| `GRPC_TLS_CA` | public grids | PEM file trusting orchestrator cert |
| `DISTRIBAI_BLOCK_ON_BENCHMARK` | optional | `true` to block tasks until benchmark completes |

Legacy docs mentioning `DISTRIBAI_ORCHESTRATOR_HOST` or `DISTRIBAI_STATE_DIR` are outdated. Use `ORCHESTRATOR_URL` and `STATE_DIR`.

## Quick start (persistent worker)

```bash
pip install -r requirements-worker.txt

export ORCHESTRATOR_URL=your-host.example.com:50051
export ADMIN_URL=http://your-host.example.com:8766
export GRPC_USE_TLS=true
export GRPC_TLS_CA=/path/to/ca.pem
export DISTRIBAI_INVITE_CODE=your-invite

python -m worker.src.daemon.run
```

Verify on the operator host with `GET /admin/nodes`; your node should appear.

## When the orchestrator is offline

Workers back off and reconnect automatically. Jobs stay queued in the operator's SQLite until the orchestrator returns. Contributors do not need the backend; they only need the join kit endpoints to come back online.

## Related docs

- Operator setup: [operator-join-checklist.md](../runbooks/operator-join-checklist.md)
- Colab / Kaggle: [ephemeral-compute-colab-kaggle.md](ephemeral-compute-colab-kaggle.md)
- TLS: [tls-and-mtls.md](tls-and-mtls.md)
- Deployment: [deployment.md](../runbooks/deployment.md)
