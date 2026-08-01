# Five-minute training onboarding

Aim: package a folder, submit a job, poll status, and find artifacts — using only the README, `.env.example`, and the CLI.

## 1. Start the orchestrator

```bash
pip install -r requirements.txt
python -m services_python.orchestrator_grpc
```

If `ADMIN_REQUIRE_AUTH=1`, place `DISTRIBAI_ADMIN_SECRET` in `.env`.

## 2. Submit the golden template

```bash
export DISTRIBAI_ADMIN_SECRET=your-secret
python -m scripts.cli.distribai_cli submit ./examples/golden_template --steps 3
# Or from a recipe file:
python -m scripts.cli.distribai_cli submit --recipe examples/job_recipe.example.json
```

The CLI emits `job_id` plus the package SHA-256 fingerprint.

## 3. Poll job status

```bash
curl -H "Authorization: Bearer $DISTRIBAI_ADMIN_SECRET" http://127.0.0.1:8766/admin/jobs/<job_id>
```

Queued jobs may include `queue_blockers` when no workers can accept work. Failures expose a stable `failure_code`.

## 4. Locate artifacts

```bash
curl -H "Authorization: Bearer $DISTRIBAI_ADMIN_SECRET" \
  http://127.0.0.1:8766/admin/jobs/<job_id>/artifacts
```

Bundles land under `runtime/bundles/{task_id}.tar.gz` unless `DISTRIBAI_BUNDLE_DIR` overrides the path.

## Safety note

Script bytes remain on the worker execution path. Operators review metadata and hashes via admin JSON; bundles are not copied to operator machines unless someone downloads them deliberately.

Notebooks: place ``run.ipynb`` at the bundle root (code cells become ``run.py``
on the worker, or the CLI materializes ``run.py`` at submit time). You can still
export to ``.py`` manually if you prefer.

See also: [node-user-guide.md](node-user-guide.md), [endpoints.md](../api/endpoints.md).
