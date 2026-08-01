# Changelog

All notable changes to this project are documented here.

<p align="center" style="line-height:1.55;"><small>
  <a href="README.md"><strong>README</strong></a> &nbsp;|&nbsp; <span>(overview)</span><br/>
  <a href="TODO.md"><strong>TODO</strong></a> &nbsp;|&nbsp; <span>(backlog)</span><br/>
  <a href="docs/README.md"><strong>Documentation index</strong></a> &nbsp;|&nbsp; <span>(all docs)</span><br/>
  <a href="docs/runbooks/deployment.md"><strong>Deploy runbook</strong></a> &nbsp;|&nbsp; <span>(smoke)</span>
</small></p>

## Unreleased

- **CLI & TUI:** consolidated the two conflicting `distribai` CLI implementations into one extensible surface (`scripts/cli/distribai_cli.py`) backed by shared `AdminAPIClient`/`ManagedProcess`/identity helpers; added a Textual terminal dashboard (`distribai tui` / `distribai-tui`) with Overview/Nodes/Jobs/Credits/Settings/Logs tabs. `distribai`, `distribai-cli`, and `distribai-tui` console scripts now register correctly via `pip install -e .`.
- **Fix:** moved the interactive packaging wizard from repo-root `setup.py` to `scripts/packaging/setup_wizard.py` — a file literally named `setup.py` is always executed by setuptools' PEP 517 backend as a legacy distutils entry point (e.g. with `egg_info`), which broke `pip install -e .` / `pip install .` / `pip wheel .` for everyone regardless of `pyproject.toml`'s declarative metadata.
- Documentation: rewrote README intro (removed release-style copy), consolidated nav styling across Markdown entry points, rebuilt `TODO.md` for readability, expanded `.env.example` with clearer optional-vs-required semantics, softened `setup.py` packaging wording, synced version badge with packaging metadata (`0.9.0`).
- **Repo layout:** `scripts/` reorganized (`cli`, `dev`, `ci`, `maintenance`, `packaging`, `publish`); dashboard static split into `worker/src/dashboard/static/{node,orch,shared}/` with matching Express mounts; root `PRODUCTION_DEPLOYMENT.md` merged into [docs/runbooks/deployment.md](docs/runbooks/deployment.md).

- Extended `TaskAssign` over gRPC with optional `script_package`, execution paradigm fields, and `distributed_env_json`.
- Job distributor sends `ServerMessage(assign=TaskAssign(...))` (protobuf) to worker queues instead of JSON dicts.
- Workers execute non-empty `script_package` tarballs via `ScriptRunner`, merging environment variables from `distributed_env_json`.
- Scheduler emits explicit `legacy_builtin` assigns for built-in training tasks.
- Submission API performs lightweight AST validation on `script_content` with structured errors and suggestions.
- Admin stubs under `/api/admin/*` return HTTP 501 instead of fake success payloads.
- `GET /admin/update-url` uses full update discovery (env + optional `UpdateService` probe).
- `UpdateService.install_update` installs `.whl` via pip or merges `.zip`/`.tar.gz` into `DISTRIBAI_INSTALL_ROOT`.
- Trusted submitters: SQLite table plus `/admin/trust/submitters` list/add/remove endpoints.
- README roadmap table for non-PyTorch frameworks (JAX, TensorFlow, MLX, ONNX Runtime).
- Node GUI: removed misleading “CUDA download” that treated a PyTorch wheel as a zip runtime.
- Repository-wide Ruff cleanup for CI (`ruff check --fix`, `ruff format`). Stray root scripts excluded via `[tool.ruff] exclude` in `pyproject.toml`.
- `worker/src/distribai_proto/distribai_pb2_grpc.py`: use `from . import distribai_pb2` so imports work under `worker.src.distribai_proto` (regenerating stubs may revert this line).
- Boss gate: `npm run verify:boss` runs full-repo Ruff + phase-contract pytest (`scripts/ci/boss_gate.cjs`).
- Dashboard landmarks: `<nav role="navigation" aria-label="Primary">`, `<main role="main" aria-label="Application content">`.
