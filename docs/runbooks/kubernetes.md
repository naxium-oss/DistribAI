# Kubernetes / Helm packaging

Experimental fleet chart for DistribAI. Path: [`deploy/helm/distribai/`](../../deploy/helm/distribai/).

This is **not** a turnkey production install. You must supply a container image that contains the repo (or an equivalent install of `services_python` + `worker`), secrets, and networking. There is no published public image from this repository.

## Install sketch

```bash
# From repo root — dry-run template render
helm template distribai ./deploy/helm/distribai \
  --set image.repository=my.registry/distribai \
  --set image.tag=0.9.0

# Install into a namespace (after building/pushing your image)
helm upgrade --install distribai ./deploy/helm/distribai \
  --namespace distribai --create-namespace \
  --set image.repository=my.registry/distribai \
  --set image.tag=0.9.0 \
  --set secrets.JWT_SECRET='…' \
  --set secrets.SIGNING_KEY='…' \
  --set secrets.DISTRIBAI_ADMIN_SECRET='…'
```

Chart pieces:

| Resource | Role |
|----------|------|
| Orchestrator `Deployment` + `Service` | gRPC + admin HTTP |
| Worker `DaemonSet` (default) or `Deployment` | Node daemons |
| `ConfigMap` | Non-secret env from `values.config` |
| Optional `Secret` | JWT / signing / admin bearer when set |
| PVC (optional) | SQLite directory for a single orchestrator replica |

## Honest operational limits

### SQLite is single-owner

The default orchestrator store is SQLite under `runtime/db/`. A PVC with `ReadWriteOnce` plus **`orchestrator.replicaCount: 1`** is the supported chart default.

Do **not** scale the orchestrator Deployment above one replica while SQLite is the authority. Concurrent writers on the same DB file cause corruption and lock storms.

### Redis for multi-replica

Multi-instance orchestrator processes need shared coordination. Set `config.REDIS_URL` (for example `redis://redis:6379/0`) and plan a separate Redis (or compatible) service before raising `orchestrator.replicaCount`. Even with Redis, review which state still lives in SQLite and whether you need an external database migration path — this chart does not invent one.

### Workers

Default `worker.mode: DaemonSet` schedules one worker pod per cluster node. Switch to `worker.mode: Deployment` with `worker.replicaCount` when you want a fixed pool (for example GPU node pools labeled separately — add nodeSelectors in a values override; not wired by default).

Workers dial the in-cluster Service name `<release>-distribai-orchestrator` on the gRPC port.

### Images and dashboards

The chart runs `python -m services_python.orchestrator_grpc` and `python -m worker.src.daemon.run`. Express dashboards (`client/server.js`, `client/orchestrator-server.js`) are not included; expose admin HTTP via Ingress if operators need the HTTP API, and run UI processes separately if required.

## Related docs

- [deployment.md](deployment.md) — secrets, TLS, SQLite hygiene on VMs/systemd
- [ports.md](ports.md) — port conventions
- [AGENTS.md](../../AGENTS.md) — no-mock production stack rule
