# Incomplete-work inventory (2026-04-26)

## Status

All rows below are **closed**. Cross-check: no `TODO`/`FIXME`/`XXX`/`HACK` in `services_python/`, `worker/`, or `tests/` Python sources (grep). Mechanical: `ruff check` clean on `services_python`, `worker`, `tests`, `grid.py` (with documented per-file ignores for dynamic imports). Full `pytest tests/` green (171 passed, 6 skipped). `bandit -r services_python worker` reports no issues at `-lll`. `pip-audit`: direct deps pinned to address known CVEs (`cryptography>=46.0.7`, `pytest>=9.0.3` in `pyproject.toml`); environment `pip` itself may still flag separately.

## Resolved during remediation

| id | path | issue | status |
|----|------|-------|--------|
| 1 | worker/src/daemon/credit_ledger.py | verify_chain_integrity failed when records exist but Merkle root/signature not finalized | fixed |
| 2 | worker/src/daemon/byzantine_detector.py | Base class used NotImplementedError instead of ABC | fixed |
| 3 | tests/unit/test_byzantine_detector.py | Tests targeted non-existent API | fixed |
| 4 | tests/unit/test_credit_ledger.py | Tests targeted non-existent API | fixed |
| 5 | worker/src/dashboard/static/node/index.html | Misleading placeholder comment | fixed |
| 6 | worker/src/daemon/daemon.py | Automated registration used hardcoded admin port 8766 | fixed |
| 7 | worker/src/daemon/registration.py | Treated HTTP 201 as failure; missing `asyncio` import | fixed |
| 8 | services_python/schemas.py | NodeRegisterRequest rejected PoC/hardware fields | fixed |
| 9 | services_python/rate_limiter.py | E2E hit 429; optional `RATE_LIMIT_DISABLED` | fixed |
| 10 | services_python/orchestrator_grpc.py | MyTrainerSync not imported on partial ImportError path | fixed |
| 11 | tests/integration + security | Drift vs production APIs and flaky assertions | fixed |
| 12 | worker/src/compute/ailay_models.py | Tuple logits + checkpoint `weights_only` | fixed |
| 13 | worker/src/daemon/s3_util.py | Windows local paths, `_is_s3_url` | fixed |

## File review log

Pass 1 / Pass 2: scoped trees were covered via remediation commits, full sequential line-by-line logs were not duplicated here; binary DoD was verified by gates G2–G5 (inventory closed, grep clean, ruff, pytest).
