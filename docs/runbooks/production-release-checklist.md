# Production release checklist

Use this list immediately before deploying **`review-security-patches-and-updates-may-2026`** (or after merge to `main`).

## Local gates (required)

Run from repo root with `venv` activated and `npm install` done:

```bash
npm run verify:production   # ruff + full pytest + Playwright (14 tests)
npm run test:newcomer       # preflight + golden_template rehearsal
python tools/verify_setup.py   # expect 34/34
```

All three must exit **0** with **0 skipped** pytest tests.

On **Windows**, pytest-xdist defaults to **2** workers (`scripts/ci/run_pytest_fast.cjs`) to avoid `WinError 10055` socket buffer exhaustion. Override with `DISTRIBAI_PYTEST_WORKERS` if needed.

## Secrets (production)

Copy [`.env.example`](../../.env.example) to `.env` on the orchestrator host. Set at minimum:

| Variable | Notes |
|----------|--------|
| `JWT_SECRET` | Strong random value |
| `DISTRIBAI_ADMIN_SECRET` | Admin Bearer; required when `ADMIN_REQUIRE_AUTH=1` or non-loopback admin bind |
| `SIGNING_KEY` | Credit ledger signing |
| `REDIS_URL` | Required for multi-instance orchestrator |
| `S3_BUCKET_NAME`, `AWS_*` | Blob handoff when using S3 |
| `ALLOWED_BLOB_HOSTS` | Comma hosts for HTTPS gradient URLs |

See [deployment.md](deployment.md) for TLS, ports, and SQLite single-owner rules.

## Deploy order

1. Orchestrator: `python -m services_python.orchestrator_grpc`
2. Worker nodes: `python -m worker.src.daemon.run`
3. Dashboards (optional): `node client/server.js` (user), `node client/orchestrator-server.js` (operator)

## Post-deploy smoke

- `GET /admin/health` returns 200 (with Bearer if auth enforced)
- One node registers over gRPC and appears in `/admin/nodes`
- Submit a short job via admin API or CLI; confirm `success` status

## Git

- Commits on the security branch must **not** use `[Cursor]` prefixes or `Co-authored-by: Cursor`.
- After history rewrite, push with: `git push --force-with-lease origin <branch>`
- Open PR to `main`; CI runs on `pull_request` per `.github/workflows/ci/ci.yml`.

## Known phase-2 (non-blocking)

- `services_python/db_manager.py` size — split when persistence changes next land
- `worker/src/dashboard/static/node/index.html` — legacy preview; production uses `dashboard.html`
