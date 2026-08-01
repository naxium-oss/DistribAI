# Operator join checklist (boss / single orchestrator)

Use this before handing contributors a join kit. Backend source stays on your machine; contributors only get worker binaries or the public worker tree.

## 1. Orchestrator (private)

- [ ] Run from private checkout or **admin PyInstaller binary** — never publish `services_python/` to the public mirror
- [ ] Stable secrets in `.env`: `JWT_SECRET`, `SIGNING_KEY`, `DISTRIBAI_ADMIN_SECRET` ([`.env.example`](../../.env.example))
- [ ] `GET /admin/health` returns `ok: true`

## 2. Public exposure (if contributors are not on LAN)

- [ ] `ADMIN_HOST` bound appropriately; HTTPS reverse proxy in front of admin API ([deployment.md](deployment.md))
- [ ] `GRPC_USE_TLS=true` with valid cert; contributors get `GRPC_TLS_CA` PEM
- [ ] `ADMIN_REQUIRE_AUTH=1` and strong `DISTRIBAI_ADMIN_SECRET` on dashboards if used
- [ ] `REGISTRATION_REQUIRE_POC=true` for open grids ([`.env.example`](../../.env.example))
- [ ] Per-team `DISTRIBAI_INVITE_CODE` values issued to contributors

## 3. Public releases (worker only)

- [ ] GitHub Release contains **node** artifacts only (`distribai-node`, not admin)
- [ ] Build via `python scripts/packaging/bundle.py node` on release CI ([packaging.md](../guides/packaging.md))
- [ ] `version.json` / manifest points to node downloads ([update-hosting.md](../guides/update-hosting.md))
- [ ] `publish_public_grid.py` dry-run: confirm `services_python/` is **not** in mirror paths

## 4. Join kit for contributors

Publish (wiki / email / pinned notebook):

- [ ] `ORCHESTRATOR_URL`
- [ ] `ADMIN_URL` (HTTPS preferred)
- [ ] `GRPC_TLS_CA` file or download link
- [ ] `DISTRIBAI_INVITE_CODE` (per team if using invites)
- [ ] Link to node release **or** public repo + `requirements-worker.txt`
- [ ] Link to [Colab template](../../examples/colab/join_grid.ipynb) for notebook users

## 5. Smoke test before announcing

- [ ] Home machine worker joins with join kit
- [ ] Colab template run → node in `/admin/nodes`
- [ ] Submit script job → worker logs `backend_used`
- [ ] Stop orchestrator → workers backoff; restart → queue drains

## 6. Rate limits (many Colab/Kaggle users)

- [ ] Review [`services_python/rate_limiter.py`](../../services_python/rate_limiter.py) defaults
- [ ] Monitor `/admin/nodes` for registration spikes

## Related

- [contributor-join-kit.md](../guides/contributor-join-kit.md)
- [ephemeral-compute-colab-kaggle.md](../guides/ephemeral-compute-colab-kaggle.md)
- [beta-preprod-checklist.md](beta-preprod-checklist.md)
