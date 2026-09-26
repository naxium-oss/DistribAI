# MyTrainer bundled subtree

> **Deprecated.** This page documents the Python-era stack under `legacy/`.
> It receives bug fixes only. For current documentation start at
> [`docs/README.md`](../../../docs/README.md), and for the mapping from a
> Python module to its C++ replacement see
> [`docs/from-python-to-cpp.md`](../../../docs/from-python-to-cpp.md).

DistribAI integrates with [MyTrainer](https://github.com/) via `external/mytrainer`. The orchestrator admin route `POST /api/admin/mytrainer/sync` reads architecture configs from that directory.

## Checkout

If the directory is empty after clone:

```bash
git clone <mytrainer-repository-url> external/mytrainer
```

Or, when the repo publishes a git submodule entry:

```bash
git submodule update --init --recursive external/mytrainer
```

## Verify locally

```bash
python scripts/ci/verify_mytrainer_submodule.py
# Fail CI/release builds that require the subtree:
python scripts/ci/verify_mytrainer_submodule.py --require
```

A healthy tree includes `external/mytrainer/configs/grid_architectures.json` (used by `services_python/mytrainer_sync.py`).

## Environment override

Set `MYTRAINER_PATH` to point at a non-default checkout; the sync handler resolves `external/mytrainer` under the repo root when unset.
