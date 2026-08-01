# DistribAI Agent Configuration

Format Python with PEP 8. Match the style already present in files you touch.

**Living backlog:** [TODO.md](TODO.md) — roadmap gaps, hygiene checks, accessibility passes. Deferred work for the 2026-08 remake sits in the top section of that file.

---

## Repository map

| Area | Role |
|------|------|
| [`services_python/`](services_python/) | Orchestrator: gRPC entrypoints, admin HTTP API, scheduler, credits/ledger hooks, monitoring; model families in `architecture_config.py` (`decoder_transformer`, `gru`, `gated_conv`, `moe_decoder`, `lstm`, `resnet_lm`, `hybrid_attn_rnn`, `dense_ffn`) |
| [`worker/`](worker/) | Node daemon; static UI under **`worker/src/dashboard/static/{node,orch,shared}/`**, benchmarks, compute backends |
| [`client/`](client/) | Node.js dashboards and proxies (contributor vs operator surfaces, separate ports/origins) |
| [`proto/`](proto/) | `distribai.proto`; regenerate Python stubs into `worker/src/distribai_proto/` when RPC contracts change |
| [`scripts/`](scripts/) | Organized tooling: CLI ([`scripts/cli`](scripts/cli)), dev previews + subprocess simulator ([`scripts/dev`](scripts/dev)), CI helpers ([`scripts/ci`](scripts/ci)), maintenance patches ([`scripts/maintenance`](scripts/maintenance)), packaging ([`scripts/packaging`](scripts/packaging)), public mirror publisher ([`scripts/publish`](scripts/publish)) |
| [`tests/`](tests/) | All automated tests (`unit`, `integration`, `e2e`, `security`, `performance`, `chaos`) |
| [`tools/`](tools/) | Dev harness utilities (`simulate_grid.py`, worker launch helpers) |
| [`specs/`](specs/) | PyInstaller `.spec` files (canonical; do not duplicate at repo root) |
| [`sdk/python/`](sdk/python/) | Published Python client package |
| [`runtime/db/schema.sql`](runtime/db/schema.sql) | SQLite schema source of truth (DB files themselves are **not** committed) |
| [`external/mytrainer`](external/mytrainer) | Bundled MyTrainer tree — see [docs/guides/mytrainer-submodule.md](docs/guides/mytrainer-submodule.md); verify with `npm run verify:submodule` |

---

## WARNING: No fake distributed stack

Production paths must run **real** orchestrator and worker logic:

- Do **not** ship mock orchestrators, mock workers, in-memory-only substitutes for persistence where SQLite/Redis is required for correctness, or placeholder "stub" services on code paths users rely on.

If you find mock implementations posing as production code, treat it as a bug. Dashboard/proxy URL mismatches that return empty or wrong shapes count as the same class of defect.

### Harnesses (real stack, one machine)

Official harness entry points (all use **real** orchestrator/worker code):

| Path | What it does |
|------|----------------|
| [`tests/e2e/test_e2e.py`](tests/e2e/test_e2e.py) | Pytest E2E harness |
| [`tools/simulate_grid.py`](tools/simulate_grid.py) | **In-process** threaded harness (orchestrator + `WorkerDaemon`; includes security test variants) |
| [`scripts/dev/simulate_grid_cli.py`](scripts/dev/simulate_grid_cli.py) | **Subprocess** CLI (`python -m scripts.dev.simulate_grid_cli`): spawns `python -m services_python.orchestrator_grpc` and `python -m worker.src.daemon.run` (shell-friendly) |

PyInstaller sources: [`specs/`](specs/) only (duplicate root `*.spec` files were removed).

**Pre-commit:** Optional local hook runners were removed (`.pre-commit-config.yaml` deleted). Use **Ruff**, **pytest**, and **CI workflows** as the source of truth; reintroduce pre-commit only if the team wants unified git hooks.

### Unit tests and `unittest.mock`

Some **unit** tests use `unittest.mock` (patching network I/O, `torch.cuda`, small collaborators). That is allowed **when it does not replace the orchestrator or WorkerDaemon as a whole**. Prefer integration/e2e tests when asserting cross-service contracts. Track broader cleanup in [TODO.md](TODO.md).

---

## Environment variables

**Committed template:** [`.env.example`](.env.example) — copy to `.env` for local overrides (gitignored).

Missing variables are fine for local dev; the stack chooses secure defaults where the code documents them.

### Optional / dev defaults

- **`JWT_SECRET`**, **`SIGNING_KEY`**: auto-generated per process if unset (`secrets.token_urlsafe`).
- **`REDIS_URL`**: omitted → in-memory fallbacks (not for production multi-instance).
- **`S3_*`**: omitted → S3 features skipped with warnings.

### Production expectations

Set strong **`JWT_SECRET`**, **`REDIS_URL`**, **`SIGNING_KEY`**, and **`S3_*`** as appropriate. Document GitHub release URLs per [docs/guides/github-releases-setup.md](docs/guides/github-releases-setup.md). Never commit `.env`; keep secrets only in your local `.env` copied from `.env.example`.

---

## Local runtime: SQLite

- Orchestrator SQLite files live under **`runtime/db/`** during runs (often named with ports, e.g. `distribai-<grpc>-<admin>.db`).
- **Do not commit** `*.db`, `*.db-wal`, or `*.db-shm`; they are gitignored. Schema lives in **`runtime/db/schema.sql`**.
- Keep **zero to three** active DB files locally if you care about disk hygiene; delete stale test databases after integration runs. All other `runtime/**` paths are local-only (see `.gitignore`; only `runtime/db/schema.sql` is tracked).

### What SQLite stores (schema excerpt)

Orchestrator state is modeled in **`runtime/db/schema.sql`**. Typical tables:

- **`active_nodes`** — registrations, JWT/session material, heartbeat timestamps, benchmarks, utilization counters, reliability score.
- **`jobs`** — job queue metadata (model names, tiers, lifecycle timestamps, pointers to latest tasks, submitter attribution).
- **`tasks`** — per-assignment execution records (`assignee_node_id`, blob URLs for weights/gradients, attempts, sandbox output summaries).
- **`credit_ledger`** — append-only-ish ledger rows with cryptographic chaining fields (`tx_hash`, `prev_hash`).
- **`vote_transactions`** — credits pledged per voting round keyed by `(job_id, voter_id)` style identifiers.

Production deployments may attach additional migrations in code paths; reconcile runtime tables against `services_python/` when auditing.

---

## Dashboard static HTML

Contributor HTML lives under **`worker/src/dashboard/static/node/`**, operator dashboards under **`.../orch/`**, and CSS/JS under **`.../shared/`**. Both Express servers mount **`/shared/`** statically; orch also mounts **`/node/`** so operator pages can link to `/node/admin.html`. Inside HTML templates prefer **root-absolute** asset URLs (`href="/dashboard.html"`, `src="/shared/scripts.js"`) so files keep working regardless of subdirectory depth.

---

## How to run (quick start)

```bash
python -m services_python.orchestrator_grpc
python -m worker.src.daemon.run
```

On Windows with a venv:

```powershell
.\venv\Scripts\python.exe -m services_python.orchestrator_grpc
.\venv\Scripts\python.exe -m worker.src.daemon.run
```

Dashboard ports are described in [README.md](README.md) (contributor vs operator separation).

---

## ALWAYS INSTALL ALL DEPENDENCIES

Keep **[`requirements.txt`](requirements.txt)** aligned with **[`pyproject.toml`](pyproject.toml)** core pins where duplicated (`cryptography`, `pytest`, etc.). CUDA installs use **[`requirements-cuda.txt`](requirements-cuda.txt)** (protobuf/grpc bounds match CPU file).

Install immediately when starting work so skipped tests do not hide regressions:

```bash
pip install -r requirements.txt
# NVIDIA CUDA builds only:
# pip install -r requirements-cuda.txt
```

Core libraries include **torch**, **numpy**, **grpcio**, **psutil**, **aiohttp** (see [`requirements.txt`](requirements.txt)); optional GUI (**pywebview**) may fail on some platforms — note skips explicitly. Optional Node tooling for dashboards/UI tests: `npm install` in repo root ([`package.json`](package.json)).

Verification snippet:

```python
import torch
import numpy
import grpc
import psutil
import aiohttp

print("Core deps available!")
```

---

## Test creation and layout

1. **Location:** Add tests only under [`tests/`](tests/). Root-level `test_*.py` is disallowed — [`pyproject.toml`](pyproject.toml) sets `testpaths = ["tests"]`.
2. **Markers:** Use `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.security`, `@pytest.mark.chaos`, `@pytest.mark.unit` consistently ([`pyproject.toml`](pyproject.toml)).
3. **Async:** Project uses `asyncio_mode = "auto"` with module-scoped loop defaults — follow existing fixtures in [`tests/conftest.py`](tests/conftest.py).
4. **Naming:** `test_<behavior>` modules alongside peers (`tests/unit/test_*.py`, etc.).
5. **Harness parity:** For orchestrator/worker behavior, prefer [`tests/integration/`](tests/integration/) or [`tests/e2e/`](tests/e2e/) over isolated mocks.

Coverage expectations: [`tool.coverage.report`] `fail_under = 100` in `pyproject.toml` (not enforced by `npm run verify:boss`; run `pytest --cov` locally when asserting coverage).

---

## PR checklist

1. Install Python deps from [`requirements.txt`](requirements.txt) (use [`requirements-cuda.txt`](requirements-cuda.txt) on NVIDIA CUDA hosts); `npm install` when touching dashboards / Playwright.
2. `ruff check` on touched Python (respect [`pyproject.toml`](pyproject.toml) excludes).
3. `npm test` — unit+security, parallel, no skips, typically under 20s; `npm run test:all` — full Python suite (~35s with fast mode); `npm run test:ui` — Playwright (safe teardown via `scripts/ci/kill_playwright_servers.cjs` — **never** `Stop-Process` all `chrome`/`chromium`/`msedge`).
4. No **`__pycache__`**, **`.pyc`**, **`.env`**, or **`runtime/db/*.db`** in commits.
5. Scan [TODO.md](TODO.md) for items relevant to your change (version drift, submodule, accessibility backlog, 2026-08 remake deferrals).

---

## Git hygiene

- **Commit and PR attribution:** All commits and pull requests are **human-authored** (EnderchefCoder / project maintainers). **Never** prefix messages with `[Cursor]` or similar AI watermarks. **Never** add `Co-authored-by: Cursor` or other agent co-author trailers. Use conventional subjects only (e.g. `fix(security): …`, `test: …`).
- To scrub historical messages on a branch: `scripts/maintenance/strip_commit_watermarks.py` (used with `git filter-branch --msg-filter` and `sed` equivalents).
- Cursor may inject `Co-authored-by: Cursor` when the agent runs `git commit`; use `python scripts/maintenance/commit_without_coauthor.py -F msg.txt` after staging, or disable agent co-author attribution in Cursor settings.
- `.gitignore` covers bytecode, caches, `dist/`, local databases, `.env`, Playwright artifacts.
- After cloning, run tests locally — stale SQLite files may accumulate; delete extras under `runtime/db/` when finished debugging.

## Playwright UI tests — safe cleanup

`npm run test:ui` wraps Playwright and runs [scripts/ci/kill_playwright_servers.cjs](scripts/ci/kill_playwright_servers.cjs) afterward.

**Allowed:** stop processes listening on `PLAYWRIGHT_PORT` (default `3210`) and `node` running `client/server.js` for the test webServer.

**Forbidden:** killing every Chrome/Chromium/Edge process (`Get-Process chrome`, `Stop-Process -Name chromium`, etc.) — that closes the user's personal browser.

Manual cleanup after a stuck run:

```bash
npm run test:ui:kill
```
