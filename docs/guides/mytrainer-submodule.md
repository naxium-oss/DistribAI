# MyTrainer bundled subtree

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
