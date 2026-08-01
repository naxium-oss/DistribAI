# docs/assets

Bundled **non-code reference** files (do not confuse with dashboard static assets under `worker/src/dashboard/static/`).

| Path | Note |
|------|------|
| [`screenshots/`](screenshots/) | README gallery (JPEG). Regenerate with `node scripts/dev/capture_readme_shots.cjs` then compress (see script footer / PIL resize). |
| `subjective_assessment.json` | Snapshot / metadata (if present) |
| `scorecard.png` | Primary scorecard image |
| `scorecard-secondary.png` | Secondary scorecard image |

Prefer keeping large generated artifacts **out of git**; compress screenshots before commit (target ~50–120 KB each).
