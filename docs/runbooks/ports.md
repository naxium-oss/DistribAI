# Ports and bind addresses

Canonical defaults for local development. Production should document overrides in your deployment manifest; keep **orchestrator admin**, **gRPC**, and **dashboard proxy targets** aligned.

| Service | Process | Env var(s) | Default | Notes |
|---------|---------|------------|---------|-------|
| Orchestrator gRPC | `python -m services_python.orchestrator_grpc` | `GRPC_PORT` | `50051` | Worker `WorkerDaemon(orchestrator_url=host:GRPC_PORT)` |
| Orchestrator admin HTTP | same | `ADMIN_PORT`, `ADMIN_HOST` | `8766`, `127.0.0.1` | REST + SSE; Bearer when auth enforced |
| Contributor dashboard | `node client/server.js` | `PORT` | `3000` | Discovers admin via `ORCHESTRATOR_ADMIN_URL` or port scan |
| Operator dashboard | `node client/orchestrator-server.js` | `ORCH_PORT` | `3212` | Loopback by default |
| Contributor admin proxy | `client/server.js` | `ORCHESTRATOR_ADMIN_URL` | `http://127.0.0.1:8766` | Must match orchestrator `ADMIN_HOST`:`ADMIN_PORT` |
| Operator admin proxy | `client/orchestrator-server.js` | `ADMIN_HOST`, `ADMIN_PORT` | `127.0.0.1`, `8766` | Built as `http://ADMIN_HOST:ADMIN_PORT` (ignores `ORCHESTRATOR_ADMIN_URL`) |
| Playwright UI tests | `npm run test:ui` | `PLAYWRIGHT_PORT` | `3210` | Ephemeral `client/server.js` webServer; not production |

**Teardown:** `npm run test:ui` runs [scripts/ci/kill_playwright_servers.cjs](../../scripts/ci/kill_playwright_servers.cjs) after tests. It stops only listeners on `PLAYWRIGHT_PORT` and test `node client/server.js` processes — **not** the user's Chrome/Edge/Chromium windows. Manual: `npm run test:ui:kill`.

## Related variables

- **`DISTRIBAI_API_URL`** — worker override for admin base URL (registration, benchmarks).
- **`GRPC_USE_TLS` / `GRPC_TLS_*`** — TLS for gRPC; see [deployment.md](deployment.md).
- **`CORS_ALLOWED_ORIGINS`** — browser origins allowed to call admin API from dashboards.

## Harness / CI ports

| Harness | gRPC | Admin |
|---------|------|-------|
| `tests/e2e/test_e2e.py` | `19765` | `19766` |
| `scripts/dev/simulate_grid_cli.py` | `19001` (default) | `19002` (default) |

Do not collide with production binds when running harnesses on the same host.
