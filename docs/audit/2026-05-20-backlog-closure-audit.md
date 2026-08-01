# Backlog closure audit (2026-05-20)

Timeboxed subtree pass for production gate. First-party trees read or verified via tests:

| Tree | Evidence |
|------|----------|
| `services_python/` | Admin auth, queue diagnostics, preflight, job failure codes, penetration tests |
| `worker/` | Sandbox, blob policy, static dashboards, operator banner snooze |
| `client/` | Dual Express proxies, shared static mounts |
| `scripts/` | CLI submit/recipe, mini_smoke, newcomer_test, rehearse_sandbox, CI gates |
| `tests/` | unit/integration/security/e2e/playwright; 0 skips enforced in `conftest.py` |
| `docs/` | five-minute-onboarding, ports, deployment TLS, endpoints job contract |
| `sdk/python/` | `distribai/jobs.py` admin parity |
| `proto/` | unchanged this pass; stubs under `worker/src/distribai_proto/` excluded from ruff |

Deferred product-scale items (K8s fleet packaging, 100GB dataset upload UI, multi-framework launchers) documented in closed `TODO.md` with honest phase-2 boundaries.
