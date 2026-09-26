# Operations

Running a grid people connect to. All examples use a coordinator on
`127.0.0.1:50061`.

## Start it properly

```bash
build/cpp_port/distribai_orch \
    --host 127.0.0.1 --port 50061 \
    --db runtime/db/grid.db \
    --schema runtime/db/schema.sql \
    --jobs-dir runtime/grid/jobs \
    --tasks-dir runtime/grid/tasks \
    --invite "$GRID_INVITE" \
    --token "$GRID_TOKEN" \
    --node-ttl 60 \
    --aggregate trimmed_mean
```

Keep `--host` on loopback. A tunnel or a reverse proxy should be what faces the
network, not the coordinator itself. `--invite` is what keeps strangers from
registering workers, and `--token` is what keeps them from submitting jobs;
set both on anything public.

Under systemd, the unit wants `Restart=always`, the environment variables above,
and a `WorkingDirectory` at the repo root so the relative default paths resolve.

## Publish it for free

The coordinator speaks HTTP, so any tunnel or proxy works. The bundled tool
covers the two free options:

```bash
make grid-providers                                        # what is available
make grid-expose ARGS="--provider cloudflare --port 50061 --invite $GRID_INVITE"
make grid-expose ARGS="--provider ngrok --port 50061"      # needs NGROK_AUTHTOKEN
```

Cloudflare's quick tunnel needs no account and no domain. It gives you an
`https://<random-words>.trycloudflare.com` URL, and the printed join link uses it
on port 443 with TLS, so notebook hosts that block odd ports still work. ngrok is
the same shape but needs a free authtoken.

Notes that matter in practice:

- Quick tunnel URLs change on every restart. Print the link again, or send the
  new one, after a bounce.
- The tunnel only proxies the coordinator. The dashboard is reachable through it
  too, which is convenient for read-only access and is a reason to keep the
  invite and the token strong.
- Local remains the default. Nothing is published unless you pass `--provider`.

## Capacity

| Knob | Guidance |
| --- | --- |
| Replicas per job | Start at the number of workers you expect to be idle. More replicas than workers just queue. |
| `--node-ttl` | 60s suits stable machines. On flaky notebook hosts, 90 to 120s avoids needless requeues; below 30s, a slow heartbeat window starts costing duplicate work. |
| `--cpu-sec` | Sized per replica. A worker's child is killed at this budget, and the task is reported as failed with a reason. |
| `--mem-mb` | The LibTorch trainer needs headroom for torch itself (8 GiB default). Keep the port's own engine at tens of MB. |
| Aggregate | `trimmed_mean` is the default because one bad replica should not decide the result. Use `mean` only when you trust every node. |

Heterogeneous workers are expected. A notebook GPU and a laptop CPU both run the
same trainer; the difference shows up as wall time and `steps_per_s` in the
result, not as a different code path.

## Watch it

```bash
curl -s localhost:50061/v1/health | python3 -m json.tool     # status, nodes, queued tasks
curl -s localhost:50061/v1/summary | python3 -m json.tool    # the dashboard payload
```

The dashboard at `/` is the fastest read: workers with their last heartbeat, jobs
with progress in steps, the credit ledger and the contributor leaderboard. It
polls `/v1/summary` every three seconds and stops while the tab is hidden.

Straight from SQLite, when you want history rather than a snapshot:

```bash
sqlite3 runtime/db/grid.db \
  "select status, count(*) from tasks group by status;
   select node_id, jobs_completed, jobs_failed, round(reliability_score,2)
     from active_nodes order by jobs_completed desc;"
```

## Back up and clean up

Two things hold state: `runtime/db/grid.db` (plus its `-wal` file while running)
and `runtime/grid/`. Copy both together, or checkpoint first:

```bash
sqlite3 runtime/db/grid.db "pragma wal_checkpoint(truncate);"
cp -a runtime/db/grid.db runtime/grid /backup/grid-$(date +%F)/
```

Disk grows with bundles and envelopes: a job directory keeps its inputs, one
envelope per replica, and the aggregate. Each translated model is small, but
hundreds of jobs add up. To prune finished work:

```bash
# find finished jobs older than a week and drop their stored files, keeping rows
find runtime/grid/jobs -maxdepth 1 -type d -mtime +7 -exec rm -rf {} +
find runtime/grid/tasks -name '*.env' -mtime +7 -delete
```

Rows stay for accounting; the result files are gone. Keep a backup if you need
the files.

## Drills worth running once

1. **Kill a worker mid-task.** With `--node-ttl 30`, the task returns to the
   queue within about half a minute and the next worker picks it up. The task's
   `last_error` reads `assignee went offline`. `make grid-test` covers this, and
   `make grid-regression-test` pins the requeue and takeover by name.
2. **Starve a replica.** Run a worker with `--mem-mb 64` against the LibTorch
   trainer. The child dies on address space, the task is marked failed with a
   reason, and the job still completes if another replica reported.
3. **Cancel a job.** `curl -X POST -H "X-Grid-Token: $GRID_TOKEN" `
   `localhost:50061/v1/jobs/<id>/cancel`. Queued and assigned tasks are failed,
   the job reads `cancelled`, and no credits are paid for work that did not
   finish.
4. **Stop the coordinator mid-job.** Restart it with the same `--db`. Registered
   nodes go offline after `--node-ttl`, their tasks requeue, and the job resumes
   when workers reconnect.

## Security checklist

- `--invite` set, and rotated when you print a new link.
- `--token` set; the submit command uses `X-Grid-Token`.
- Coordinator bound to loopback, with the tunnel or proxy facing the network.
- Workers run as an unprivileged user, with `--mem-mb`, `--cpu-sec` and
  `--fsize-mb` set to what a replica actually needs.
- Trainer children inherit only `PATH` and `TMPDIR` from the worker, and run in
  their own PID, user and mount namespaces where the kernel allows.
- Read the ledger before trusting it. `tx_hash` chains rows, so a tampered row
  breaks the next link, but it is CRC-32 rather than a signature.

## Upgrades

Binaries are self-contained. To upgrade a running grid:

1. Pause submissions, wait for running jobs to finish or cancel them.
2. Stop the coordinator, keep `--db` and the grid directory.
3. Rebuild (`make`, `make grid`, `make torch`).
4. Start the coordinator again. The schema is applied with `CREATE TABLE IF NOT
   EXISTS`, so an older database keeps working. Columns added since a database
   was created are patched in on start: the pass records each existing column
   and runs `ALTER TABLE ... ADD COLUMN` for the missing ones, ignoring the
   duplicate-column error. `jobs.aggregate`, added for per-job aggregation, is
   one of these.
5. Restart workers. They re-register and rotate their session tokens.

Rolling worker upgrades need no coordination: a worker that reconnects with the
same `--node-id` keeps its counters. On a link that drops often, raise the
worker's `--register-wait` so a registration attempt rides out a short outage
instead of giving up. The worker only retries transport failures; a refused
invite or a rejected node id is final and reported as such.

## Verify a change before you ship it

The gates in [Testing](testing.md) run the whole control plane on loopback, so
they catch the mistakes an operator would otherwise find in production. Before
a release:

```bash
make test port-check     # port gates plus the bench report
make suite-quick         # suite subset with sanitizers
make grid-all            # every grid and trainer gate
```

`make port-check` writes its evidence to
`runtime/baselines/cpp_port_check_report.md`. Read that file rather than trusting
a green summary: it lists each gate and the checks that ran.

## Troubleshooting index

| Symptom | Likely cause |
| --- | --- |
| `tasks_queued` stays high with healthy nodes | Workers cannot reach the coordinator. Check their URL, TLS scheme and the invite code. |
| `nodes_online` is 0 but workers run | Wrong URL, or the coordinator restarted and the workers hold stale tokens. They re-register on the next 401. |
| Job completes with `contributors: 1` out of 2 | One replica failed; read `replicas_detail` in `result.json` for the error. |
| `aggregate_grad_len: 0` | No usable envelope arrived, so there was nothing to aggregate. Each replica's error is in `replicas_detail`. |
| Coordinator cannot bind | Another process holds the port. `ss -ltnp | grep 50061`. |
| `dashboard asset not found` | Start with `--web-dir tools/cpp_port/grid/web`. |
| Worker exits with `invite code required` | Pass the same `--invite` the coordinator was started with. The worker does not retry this. |
| Job detail shows `aggregate` different from what you passed | The bundle's own `aggregate` wins over the submit flag. Re-translate the job, or submit a bundle that does not pin it. |
