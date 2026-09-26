# Native grid

The C++ control plane. Two binaries, one HTTP port, SQLite for state.

```
build/cpp_port/distribai_orch     coordinator: worker API, operator API, dashboard
build/cpp_port/distribai_worker   runs one replica of a job on this machine
```

Nothing here needs Python. The coordinator serves the worker API, the operator
API and the dashboard from the same port, which is why a free Cloudflare quick
tunnel is enough to let contributors join from Colab, Kaggle, Molab or a VPS.

Build both with `make grid` (needs `libsqlite3-dev`; the worker alone needs only
libc, plus OpenSSL when you want `https://` orchestrator URLs).

---

## Run a grid

```bash
# 1) coordinator
build/cpp_port/distribai_orch \
    --host 127.0.0.1 --port 50061 \
    --db runtime/db/grid.db \
    --jobs-dir runtime/grid/jobs --tasks-dir runtime/grid/tasks \
    --invite team-alpha --token operator-secret \
    --print-join

# 2) workers (same machine, a LAN box, or a free GPU session)
build/cpp_port/distribai_worker --orchestrator http://127.0.0.1:50061 \
    --invite team-alpha --node-id my-gpu

# 3) a job, from a directory tools/trainer_translate produced
build/cpp_port/distribai_orch --db runtime/db/grid.db \
    --submit jobs/my-job --replicas 4

# 4) watch it
xdg-open http://127.0.0.1:50061/
```

Expose it for remote workers with the free tunnel tool:

```bash
make grid-expose ARGS="--provider cloudflare --port 50061 --invite team-alpha"
# then, on any other machine
build/cpp_port/distribai_worker --orchestrator https://random-words.trycloudflare.com \
    --invite team-alpha
```

---

## How a job flows

1. **Submit.** The operator points `--submit` (or `POST /v1/jobs`) at a job
   directory holding `job.json`, the TorchScript module and any data files. The
   coordinator snapshots that directory into `runtime/grid/jobs/<job_id>/bundle`
   so later edits to the source directory cannot change a running job.
2. **Queue.** One task per replica. Replicas differ by seed, which the
   coordinator writes into each task's copy of `job.json`.
3. **Claim.** A worker posts to `/v1/claim` and gets its task, the rewritten
   spec, and the bundle files as base64. No shared filesystem is involved.
4. **Run.** The worker unpacks the bundle and runs the trainer
   (`distribai_torch_train` by default) as a child process under RLIMIT_AS,
   RLIMIT_CPU, RLIMIT_FSIZE, RLIMIT_NPROC, RLIMIT_NOFILE and RLIMIT_CORE. The
   trainer writes a GradReport envelope and prints its metrics as one JSON line.
5. **Report.** The worker posts the envelope back with `/v1/result`. The
   coordinator stores it under `runtime/grid/tasks/<task_id>.env` and marks the
   task done.
6. **Aggregate.** The request that reports the last replica closes the job: it
   aggregates the gradients (mean, median or trimmed mean), writes `aggregate.env`
   and `result.json` next to the bundle, marks the job complete and pays credits
   into `credit_ledger`. Only one request can win that transition, so a result
   arriving at the same moment as the maintenance sweep cannot pay twice.

The aggregate method comes from the job's own `job.json` when it names one, and
from `--aggregate` otherwise. It is stored on the job row, so the report and the
API always agree on how the numbers were produced.

A node that stops heartbeating for `--node-ttl` seconds is marked offline and its
task goes back to the queue, up to `max_attempts` (3) per task. A job with no
surviving replica is marked failed with a reason. A worker that cannot reach the
coordinator retries registration for `--register-wait` seconds (20 by default) and
stops immediately on a refusal it actually received, such as a wrong invite code.

---

## Worker API

Every request is JSON, every response is JSON.

| Endpoint | Body | Answer |
| --- | --- | --- |
| `POST /v1/register` | `{proto, node_id, invite, hardware, benchmark}` | `session_token`, heartbeat and poll intervals, aggregate |
| `POST /v1/heartbeat` | `{node_id, token, status, current_task_id}` | `ack` |
| `POST /v1/claim` | `{node_id, token}` | `task` with `spec` and `files`, or `idle` |
| `POST /v1/result` | `{node_id, token, task_id, ok, envelope_b64, ...}` | `ack` with the new task status |
| `POST /v1/bye` | `{node_id, token}` | `ack` |

`proto` is 1. A worker that sends another value is refused at registration rather
than failing later.

`session_token` is bound to the node that registered. Every worker call must
carry the node id and its current token, and a re-registration rotates the token
so an old one stops working. A worker may only report on the task the grid handed
it: a report for a task assigned to another node is refused with 403, so a
registered stranger cannot fail or finish someone else's replica.

## Operator and read API

| Endpoint | Notes |
| --- | --- |
| `POST /v1/jobs` | submit `{job_dir, replicas, description}`. Needs `X-Grid-Token` when `--token` is set, otherwise loopback only. |
| `POST /v1/jobs/<id>/cancel` | fail the queued and assigned tasks, mark the job cancelled. Same auth as submit. |
| `GET /v1/health` | status, online nodes, queued tasks. Safe to expose. |
| `GET /v1/summary` | the dashboard payload: totals, nodes, jobs, tasks, ledger, leaderboard. |
| `GET /v1/nodes`, `GET /v1/tasks` | flat lists. |
| `GET /v1/jobs/<id>` | one job with per-replica metrics and the stored result. |
| `GET /` | the dashboard. |

Read endpoints are open on purpose: the dashboard needs them and they expose
nothing that a node id and a loss curve do not already show. Writes are gated.

---

## Dashboard

`web/` holds the operator dashboard: plain HTML, CSS and JavaScript, no build
step and no CDN, served by the coordinator itself. It polls `/v1/summary` every
few seconds, pauses while the tab is hidden, and reflows into labelled rows below
720px so it works on a phone.

```bash
build/cpp_port/distribai_orch --web-dir tools/cpp_port/grid/web   # default
```

If the assets cannot be found the coordinator still answers the JSON API and the
dashboard route explains which `--web-dir` to pass.

---

## Files on disk

| Path | Contents |
| --- | --- |
| `runtime/db/grid.db` | nodes, jobs, tasks, credits. Schema: `runtime/db/schema.sql`. |
| `runtime/grid/jobs/<id>/bundle/` | the frozen job inputs |
| `runtime/grid/jobs/<id>/submitted.json` | what was submitted, and the original spec |
| `runtime/grid/jobs/<id>/result.json` | per-replica metrics and aggregate sums |
| `runtime/grid/jobs/<id>/aggregate.env` | the aggregated gradients as a GradReport envelope |
| `runtime/grid/tasks/<task_id>.env` | one result envelope per replica |
| `runtime/grid/worker/<task_id>/` | a worker's scratch copy, including `trainer.log` |

Job ids look like `job-1790302442-852beb`: a timestamp plus three random bytes.

---

## Credits

Completed replicas earn credits in `credit_ledger`: `steps / --credit-divisor`
(1000 steps per credit by default). Rows chain by hash, so the ledger is
append-only in practice and tampering with an old row breaks the next link. A
node's reliability score moves up 0.02 per success and down 0.05 per failure,
clamped to 0 and 1.

This is a contribution ledger, not a token. Nothing here trades, and the
`tx_hash` is a CRC-32 chain over the row contents rather than a signature.

---

## Growing the grid

Same join line everywhere, because the worker only needs the URL and the invite:

| Where | How |
| --- | --- |
| This machine | `distribai_worker --orchestrator http://127.0.0.1:50061` |
| LAN GPU box | point `--host` at the LAN interface, share `http://<lan-ip>:50061` |
| Colab / Kaggle / Molab | `tools/gridlink/notebooks/*_join.ipynb` |
| VPS or burst server | run the worker under systemd, no `--once` |
| Anywhere else | `gridlink expose --provider cloudflare`, then share the link |

---

## Tests

```bash
make -C tools/cpp_port grid-test             # coordinator + two workers + one job
make -C tools/cpp_port grid-edge-test        # refusals, malformed input, restarts
make -C tools/cpp_port grid-regression-test  # the bugs that must not come back
make -C tools/cpp_port grid-security-test    # who may call what, plus leakage
make -C tools/cpp_port grid-soak-test        # many workers, many jobs, one burst
make -C tools/cpp_port usage-test            # run, cancel, restart, resume
make -C tools/cpp_port grid-torch-test       # a translated script on real workers
make -C tools/cpp_port grid-all              # all of the above
```

`tests/grid_e2e.sh` starts a real coordinator on loopback, submits a job,
registers two real workers, checks the aggregate, the ledger and the dashboard,
then kills a worker mid-task and confirms the task goes back to the queue. The
trainer in that gate is `tests/grid_fake_trainer.cpp`, a fixture that speaks the
same CLI and envelope contract as the LibTorch trainer so the gate runs without
torch.

`tests/grid_edge.sh` sends the coordinator what a broken client or a hostile
stranger would, including a database written by the previous schema version.
`tests/grid_regression.sh` names each past bug by its symptom.
`tests/grid_security.sh` checks authorization rather than input: the admin token,
the invite, sessions bound to a node, one worker blocked from another's task, and
no token appearing in any response. `tests/grid_soak.sh` submits a batch of jobs
to a crowd of workers and reconciles the queue, the ledger and the leaderboard.
`tests/real_usage.sh` runs, cancels, restarts the coordinator on the same
database, and reads the operator surfaces back.

Every one of those scripts picks a random port and refuses to start if a grid is
already answering there, so a coordinator left behind by an interrupted run cannot
answer for a new one.

`tests/grid_e2e_torch.sh` is the whole product slice with nothing stood in for:
the translator turns a plain PyTorch script into a bundle, the bundle is
submitted, and two real workers train it with the real LibTorch trainer.
