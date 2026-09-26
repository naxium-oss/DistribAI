# Legacy Python stack (deprecated)

This directory holds the Python-era codebase: the gRPC orchestrator, the worker
daemon, the Node dashboards, their tests and their docs. It was moved here
verbatim on 2026-09-24 when the C++ port under
[`../tools/cpp_port`](../tools/cpp_port/) became the product.

**Status: deprecated. Bug fixes only.**

That means:

- Bug fixes that keep it building and its tests passing are welcome.
- New features are not. Add them to the C++ stack instead.
- New deployments are not expected. The C++ coordinator
  ([`../tools/cpp_port/grid`](../tools/cpp_port/grid/)) replaces it.
- The Python train path is parked. Dispatching a training job raises
  `LegacyTrainPathDisabledError` from
  [`services_python/train_path_guard.py`](services_python/train_path_guard.py).
  The only bypass is `DISTRIBAI_ALLOW_LEGACY_TRAIN=1`, for debugging this frozen
  stack, and it is not a supported configuration.
- New documentation belongs at the repo root under [`../docs`](../docs/), not in
  `docs/` here.

Everyone working from this root keeps the original paths: packaging
(`pyproject.toml`, `requirements*.txt`), tests (`tests/`), dashboards (`client/`),
specs, deploy and infra charts. `runtime/` is a symlink to the shared
[`../runtime/`](../runtime/), so schema and baseline paths in tests stay valid.

Install editable from the repo root when you are fixing this stack:

```bash
pip install -e legacy
```

## Gates (run from this directory)

```bash
ruff check .
pytest tests/unit tests/security
mypy services_python/ worker/src/ --ignore-missing-imports --no-strict-optional
```

From the repo root, `make legacy-lint`, `make legacy-typecheck` and
`make legacy-test` run the same commands.

## Where to go instead

| Task | Use |
| --- | --- |
| Run a grid | `build/cpp_port/distribai_orch` plus `distribai_worker` |
| Train a real model | `distribai_torch_train` |
| Turn a Python trainer into a job | `tools/trainer_translate` |
| Expose the grid and share a join link | `tools/gridlink` |
| Read the architecture | [`../docs/architecture.md`](../docs/architecture.md) |
| Map a legacy module to its replacement | [`../docs/from-python-to-cpp.md`](../docs/from-python-to-cpp.md) |
