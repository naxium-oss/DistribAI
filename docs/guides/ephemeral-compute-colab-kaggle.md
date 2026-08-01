# Ephemeral compute (Colab, Kaggle, burst servers)

Google Colab, Kaggle notebooks, and short-lived cloud VMs are **burst workers** — they join while a session runs, then disappear. They use the same join kit as desktops ([contributor-join-kit.md](contributor-join-kit.md)) but with ephemeral settings.

## What works

- Headless worker: `python -m worker.src.daemon.run` (no GUI)
- Registration + PoC over `ADMIN_URL`
- Training and script jobs via gRPC
- **Subprocess** sandbox when Docker/nsjail are unavailable ([sandbox-backends.md](sandbox-backends.md))

## Limits (read before promising contributors)

| Constraint | Impact |
|------------|--------|
| Session timeout (Colab ~12h, Kaggle varies) | Node drops when runtime ends; re-run notebook to rejoin |
| Operator intermittently offline | Worker reconnect loops; jobs queue until orchestrator is back |
| Outbound network | Notebook must reach operator **gRPC port**; some hosts block non-443 — test first |
| No orchestrator on notebook | Never run `services_python.orchestrator_grpc` in Colab/Kaggle |
| Ephemeral disk | Set `DISTRIBAI_EPHEMERAL=1` — no long-lived JWT on disk |

## Recommended settings

```bash
export ORCHESTRATOR_URL=your-operator.example.com:50051
export ADMIN_URL=https://your-operator.example.com
export GRPC_USE_TLS=true
export GRPC_TLS_CA=/content/orchestrator-ca.pem   # upload PEM in notebook
export DISTRIBAI_INVITE_CODE=your-team-code
export DISTRIBAI_EPHEMERAL=1
```

## Notebook templates

- [examples/colab/join_grid.ipynb](../../examples/colab/join_grid.ipynb)
- [examples/kaggle/join_grid.ipynb](../../examples/kaggle/join_grid.ipynb)

Set the operator hostname and paste your invite code in the first cell.

## Typical session flow

1. Operator confirms orchestrator is up (`GET /admin/health`).
2. Contributor opens template, sets endpoints + invite, runs all cells.
3. Worker registers; node appears in `GET /admin/nodes`.
4. Operator submits jobs; worker executes until session ends.
5. Session ends → node disappears; contributor re-runs notebook next time.

## Checkpoints and artifacts

Notebook sessions can die without warning. For long training jobs:

- Prefer shorter job slices the operator can reschedule.
- Use script bundles that write checkpoints to operator-visible storage (S3) when configured.
- Do not assume local `/content` survives after disconnect.

## VPS / cloud server (non-ephemeral)

Same worker binary and env vars, but **omit** `DISTRIBAI_EPHEMERAL` and use `systemd` for a persistent daemon. See [contributor-join-kit.md](contributor-join-kit.md).

## If gRPC is blocked

Some notebook providers filter outbound ports. Test connectivity:

```python
import socket
host, port = "your-operator.example.com", 50051
s = socket.socket()
s.settimeout(5)
try:
    s.connect((host, port))
    print("gRPC port reachable")
except OSError as e:
    print("Blocked:", e)
finally:
    s.close()
```

If blocked, ask the operator about TLS on port 443 or a tunnel — see Phase 4 notes in the ephemeral onboarding plan (build only when proven necessary).
