# Beta worker rollout

This guide covers rolling out **real workers** against a **real orchestrator** after the `cleanup-before-beta` pass.

## Prerequisites

- Orchestrator host with stable secrets ([environment-reference.md](environment-reference.md))
- TLS material for gRPC if workers are not on loopback ([tls-and-mtls.md](tls-and-mtls.md))
- Python venv with `pip install -r requirements-worker.txt` on each worker (or use the node binary from releases)
- Optional: Docker or nsjail for script job isolation ([sandbox-backends.md](sandbox-backends.md))

See [contributor-join-kit.md](contributor-join-kit.md) for join credentials without backend source. Colab/Kaggle: [ephemeral-compute-colab-kaggle.md](ephemeral-compute-colab-kaggle.md).

## Step 1 — Orchestrator

```bash
python -m services_python.orchestrator_grpc
```

Confirm admin health: `GET /admin/health` on `ADMIN_PORT`.

## Step 2 — Worker daemon

```bash
python -m worker.src.daemon.run
```

Point worker config at orchestrator gRPC target (host:port). Registration returns a node JWT stored locally.

## Step 3 — Dashboards (optional)

```bash
node client/server.js          # contributor UI
node client/orchestrator-server.js  # operator UI
```

## Step 4 — Submit a script job

Use admin API `POST /admin/jobs` with `script_package_b64` or training hyperparams. Verify:

- Job appears in `GET /admin/jobs`
- Worker logs show sandbox `backend_used`
- Task completes or fails with structured error (no mock paths)

## Step 5 — Multi-device checklist

Complete [beta-preprod-checklist.md](../runbooks/beta-preprod-checklist.md) manual section with your fleet notes.

## Troubleshooting

- **401 on admin API:** set `Authorization: Bearer <DISTRIBAI_ADMIN_SECRET>` or use loopback without `ADMIN_REQUIRE_AUTH`.
- **TLS startup failure:** run `gen_tls_certs.py` or set `GRPC_USE_TLS=false` in dev only.
- **Script pip blocked:** expected when `DISTRIBAI_DENY_EGRESS=true`; bundle dependencies in the tarball.
