# Environment reference

Canonical template: [`.env.example`](../../.env.example). Copy to `.env` locally (gitignored).

## Orchestrator

| Variable | Required (prod) | Purpose |
|----------|-----------------|---------|
| `DISTRIBAI_ENV` | recommended | `production` enables fail-closed TLS and startup checks |
| `GRPC_PORT` | no | gRPC listen port (default 50051) |
| `ADMIN_PORT` | no | Admin HTTP port (default 8766) |
| `ADMIN_HOST` | yes (public) | Bind address; non-loopback triggers auth/TLS warnings |
| `ADMIN_REQUIRE_AUTH` | yes (public) | Force Bearer auth on `/admin/*` |
| `DISTRIBAI_ADMIN_SECRET` | yes (public) | Shared secret for admin HTTP Bearer token |
| `JWT_SECRET` | yes | Node session JWT signing |
| `SIGNING_KEY` | yes | Credit ledger chain signing |
| `GRPC_USE_TLS` | yes (public) | gRPC TLS (default `true`) |
| `GRPC_TLS_CERT` / `GRPC_TLS_KEY` | yes (public) | Server certificate paths |
| `REDIS_URL` | multi-instance | Shared rate limits / SSE (optional single-node) |

## Worker

| Variable | Purpose |
|----------|---------|
| `ORCHESTRATOR_URL` | gRPC host:port (default `localhost:50051`) |
| `ADMIN_URL` | Admin base URL for registration |
| `DISTRIBAI_API_URL` | Override admin URL |
| `DISTRIBAI_JWT_TOKEN` | Pre-issued JWT (skip registration) |
| `DISTRIBAI_INVITE_CODE` | Registration invite gate |
| `DISTRIBAI_EPHEMERAL` | `1` — temp state + in-memory auth (Colab/Kaggle) |
| `STATE_DIR` | Worker state directory |
| `DISTRIBAI_SANDBOX_BACKEND` | Force sandbox backend |
| `DISTRIBAI_NETWORK_POLICY` | Job network policy default |
| `DISTRIBAI_DENY_EGRESS` | Block pip install in script jobs |
| `DISTRIBAI_DEFAULT_OPTIMIZER` | Default training optimizer name (default `auon`) |
| `GRPC_USE_TLS` | Worker TLS to orchestrator |
| `GRPC_TLS_CA` | Trust bundle for orchestrator gRPC TLS |

See [contributor-join-kit.md](contributor-join-kit.md).

## Dashboards

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | 3000 | Contributor dashboard |
| `ORCH_PORT` | 3212 | Operator dashboard |
| `LISTEN_HOST` | 127.0.0.1 | Contributor bind |

See [ports.md](../runbooks/ports.md) for the full port matrix.
