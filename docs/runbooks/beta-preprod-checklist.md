# Beta pre-production checklist

Use this runbook on branch **`cleanup-before-beta`** before multi-device worker rollout.

## Baseline gates (record date + commit)

| Gate | Command | Expected |
|------|---------|----------|
| Boss | `npm run verify:boss` | ruff clean; unit+security pytest; 0 skipped |
| Production | `npm run verify:production` | boss + full pytest + newcomer + setup verify |
| Newcomer | `npm run test:newcomer` | onboarding script passes |
| Setup | `python tools/verify_setup.py` | core imports OK |

Record results here:

- **Branch:** `cleanup-before-beta`
- **Base:** `origin/main` @ `4956c75`
- **Date:** 2026-05-28
- **verify:boss:** PASS (after ScriptRunner sandbox + TLS unify + db split)
- **verify:production:** PASS (full pytest + Playwright UI, 2026-05-28)

## Blockers closed in this pass

- [x] **ScriptRunner → sandbox:** `worker/src/daemon/script_runner.py` uses `build_sandbox()`; results include `backend_used`.
- [x] **TLS unify:** `services_python/grpc_tls.py` is single source; `serve()` calls `configure_grpc_server()`.
- [x] **Commit helper:** `scripts/maintenance/commit_without_coauthor.py` honors `MERGE_HEAD` for two-parent merges.
- [x] **db_manager split:** `services_python/db/*` mixins + thin `db_manager.py` re-export.
- [x] **client/server modularize:** `client/lib/*`, `client/routes/*`; bootstrap in `client/server.js`.
- [x] **index.html decompose:** external `index-security.js`, `index-preview.js`, `index-dev-panel.js`.

## Operator secrets (production)

Copy [`.env.example`](../../.env.example) → `.env` and set:

- `JWT_SECRET`, `SIGNING_KEY`, `DISTRIBAI_ADMIN_SECRET`
- `GRPC_USE_TLS=true` with certs under `runtime/secrets/tls/` (see [tls-and-mtls.md](../guides/tls-and-mtls.md))
- `REDIS_URL` when running multiple orchestrator instances

## Multi-device beta smoke (manual)

1. Start orchestrator on host A with TLS + admin secret.
2. Start worker on host B pointing gRPC target at A.
3. Submit script job via admin API; confirm `backend_used` in worker logs.
4. Open contributor UI on B (`node client/server.js`); confirm registration + heartbeat.
5. Cancel in-flight script job; confirm subprocess backend terminates.

## Related docs

- [Beta worker rollout](../guides/beta-worker-rollout.md)
- [Sandbox backends](../guides/sandbox-backends.md)
- [Environment reference](../guides/environment-reference.md)
- [TLS and mTLS](../guides/tls-and-mtls.md)
