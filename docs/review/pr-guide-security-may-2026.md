# PR review guide: security patches (May 2026)

Branch: **`review-security-patches-and-updates-may-2026`** → **`main`**

## merge with main (2026-05-29)

PR is now **mergeable** with `main` at v1.2 integration. Conflict resolution:

| Area | Resolution |
|------|------------|
| Tree layout | Security branch kept (`scripts/dev/`, `.github/workflows/ci/ci.yml`, dashboard `node/orch/shared/`) |
| v1.2 features from main | DiLoCo, PowerSGD, AuON, sandbox backends, Byzantine defenses, TLS helpers, Makefile, SECURITY.md |
| Admin auth | **Both**: `DISTRIBAI_ADMIN_SECRET` middleware (security branch) **and** handler `required_kind="admin"` JWT (v1.1 from main) |
| gRPC register | v1.1 JWT bootstrap gate restored; PoC path when `REGISTRATION_REQUIRE_POC=1` |
| `/v1/nodes/register` | 403 unless `DISTRIBAI_ALLOW_INSECURE_REGISTER=1`; PoC policy checked first |
| Secret persistence | `services_python/constants.py` from main (disk-backed JWT/SIGNING_KEY) |
| TLS scripts | `scripts/dev/gen_tls_certs.py`, `scripts/dev/mint_admin_jwt.py` |

**Start review here after merge:**

1. P0 security: `admin_auth.py`, `grpc_service.py` (register JWT), `blob_url_policy.py`, `harness_policy.py`
2. v1.2 additive: `services_python/diloco.py`, `worker/src/sandbox/backends/`, `worker/src/daemon/optimizers.py`
3. Merge glue: `services_python/constants.py`, `services_python/orchestrator_grpc.py` (`_setup_grpc_tls`), `services_python/db_manager.py` (`get_node_jwt`)

**Gate:** `npm run verify:boss` PASS (ruff + unit/security pytest, 0 skipped).

## TL;DR

Twenty-nine focused commits add production security hardening without replacing the real orchestrator/worker stack:

1. **Security** — Admin Bearer auth (`DISTRIBAI_ADMIN_SECRET`), registration PoC policy, CORS/blob allowlists, gRPC TLS hooks, SSE limits, script tarball preflight + AST lint.
2. **Orchestrator** — Admin HTTP API registry, preflight/failure codes, queue diagnostics, bundle store, memory manager.
3. **Worker** — Executor/sandbox updates, OOM guard, dashboard static split (`node/`, `orch/`, `shared/`).
4. **Client** — Contributor + operator Express servers (`client/server.js`, `client/orchestrator-server.js`).
5. **Tests/CI** — Unit/integration/security/Playwright suites; `npm run verify:boss` gate; GitHub workflows under `.github/workflows/ci/`.

## Start here (review order)

| Priority | Paths | Why |
|----------|-------|-----|
| P0 | `services_python/admin_auth.py`, `registration_policy.py`, `blob_url_policy.py`, `cors_policy.py`, `grpc_tls.py` | Auth and fetch policy boundaries |
| P0 | `services_python/orchestrator_grpc.py`, `services_python/admin_api/` | Route wiring + startup refusal on insecure public bind |
| P1 | `worker/src/sandbox/`, `worker/src/daemon/executor.py` | Script jobs and blob fetch parity |
| P1 | `client/server.js`, `client/orch-proxy.js`, `client/orchestrator-server.js` | Proxy headers and static mounts |
| P2 | `tests/security/`, `tests/unit/test_admin_auth.py`, `tests/unit/test_blob_url_policy*.py` | Security regression coverage |
| P2 | `.env.example`, `docs/runbooks/deployment.md`, `docs/runbooks/ports.md` | Operator configuration |

## Mechanical vs behavioral

- **Mechanical:** `__pycache__` removal, runtime state untracked, cursor agent defs, large test file additions.
- **Behavioral:** Admin auth enforcement, blob URL allowlist on worker batch loads, script validation on admin job create, startup checks for public `ADMIN_HOST`.

## Risk areas

- **Admin auth:** Dashboards and curl need `Authorization: Bearer $DISTRIBAI_ADMIN_SECRET` when `ADMIN_REQUIRE_AUTH=1` or non-loopback `ADMIN_HOST`. JWT is **not** accepted as admin secret.
- **gRPC TLS:** Mismatch between orchestrator and worker cert env vars blocks all node streams.
- **Script jobs:** Tarball preflight + AST lint are early gates; runtime sandbox still owns execution isolation.
- **Large diff (~467 files):** Prioritize security modules and admin routes before dashboard HTML/CSS.

## Test plan

```bash
pip install -r requirements.txt
npm install
npm run verify:production    # boss + full pytest + Playwright (release gate)
npm run test:newcomer        # preflight unit + golden_template rehearsal
python tools/verify_setup.py # 34/34 component checks (repo root on PYTHONPATH)
```

Focused slice (faster):

```bash
npm run verify:boss
pytest tests/unit/test_admin_auth.py tests/unit/test_grpc_tls.py tests/unit/test_harness_policy.py -v
pytest tests/security -v
```

CI runs on PRs targeting **`main`** or **`develop`** (see `.github/workflows/ci/ci.yml`).

### Production gate (2026-05-28, branch HEAD)

| Command | Result |
|---------|--------|
| `npm run verify:production` | PASS (boss + full pytest + 14/14 Playwright) |
| `npm run test:newcomer` | PASS |
| `python tools/verify_setup.py` | PASS (34/34) |

## Known follow-ups (not blockers for this review pass)

- `worker/src/dashboard/static/node/index.html` (~7k lines) — legacy preview SPA; normal traffic uses `dashboard.html`. Decompose when Playwright/harness migrate off `?preview=1`.
- `client/server.js` (~1.6k lines) — orch JSON fetch + SSE relay use `orch-proxy.js`; further route-table consolidation optional.
- `services_python/db_manager.py` (~1.3k lines) — split by domain when next touching persistence.

## Structural cleanup (this review pass)

- `client/orch-proxy.js` — `createOrchProxy()` with `fetchOrchJson`, stream pipe, proxy helpers; tests in `tests/unit/test_orch_proxy.py`.
- `tests/unit/test_executor_batch_blob_policy.py` — worker batch URL allowlist parity.
- `docs/api/grpc.md` — proto/stub regen entry point.

## Commit map (newest first)

See `git log --oneline main..HEAD` for the full list. Grouped by theme:

- `da21240` … `09811a4` — hygiene, CI, security modules
- `50493e0` … `d3dda87` — orchestrator diagnostics + admin API
- `7cdf604` … `5e43b3c` — worker, client, SDK, proto, build
- `5b6b08c` … `fe0a396` — docs, examples, tests
- `452c98a` … `871ce88` — tooling and split-commit helpers
