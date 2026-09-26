# Quickstart

Two paths below. The first runs one job on this machine to show the pieces. The
second turns the machine into a grid others can join.

## What you need

| Tool | Why | Version used here |
| --- | --- | --- |
| `g++` | compiles the port and the LibTorch trainer | 14.2, C++20 for torch |
| `make` | builds everything | 4.4 |
| `libsqlite3-dev` | coordinator state | 3.46 |
| Python 3.12+ with `torch` | to translate a trainer and to build the LibTorch target | torch 2.14 |
| `cloudflared` | only for the free public tunnel | auto-downloads |
| `google-chrome` | only for `make screenshots` | optional |

```bash
sudo apt-get install -y g++ make libsqlite3-dev
python3 -m venv .venv && .venv/bin/pip install torch
```

## Build

```bash
make            # port binaries into build/cpp_port
make grid       # coordinator and worker
make torch      # LibTorch trainer (needs torch in .venv)
```

`make` alone is enough for the engine, sandbox and parity gates. `make grid`
adds the control plane and skips the coordinator with a note if SQLite headers
are missing.

## Translate a trainer

```bash
PYTHONPATH=tools .venv/bin/python -m trainer_translate.translate \
    my_trainer.py --out jobs/hello --steps 200
```

You get a directory:

```text
jobs/hello/job.json     optimizer, loss, steps, batch size, seed
jobs/hello/model.pt     TorchScript module captured from your script
jobs/hello/x.bin,y.bin  optional data snapshot
```

Translation fails closed. If your script uses something the port cannot express,
nothing is written, the command exits 3, and you get a notice naming the missing
feature plus what to do about it.

## Run one job on one worker

```bash
# coordinator, foreground
build/cpp_port/distribai_orch --port 50061 --db runtime/db/grid.db \
    --jobs-dir runtime/grid/jobs --tasks-dir runtime/grid/tasks

# worker, in another shell
build/cpp_port/distribai_worker --orchestrator http://127.0.0.1:50061 \
    --node-id first-node

# submit, in a third shell
build/cpp_port/distribai_orch --db runtime/db/grid.db \
    --submit jobs/hello --replicas 2
```

Then open <http://127.0.0.1:50061/>. The job moves from queued to running to
completed, and two replicas show up under the job. When it finishes you get:

```bash
cat runtime/grid/jobs/<job_id>/result.json      # per-replica metrics, aggregate sums
ls  runtime/grid/jobs/<job_id>/aggregate.env    # aggregated gradients, envelope format
```

If the coordinator is not up yet, the worker waits: it retries registration for
20 seconds by default. Pass `--register-wait 0` to try once and exit, or a larger
number on a slow link.

## Run a grid others can join

```bash
# publish the coordinator through a free Cloudflare quick tunnel
make grid-expose ARGS="--provider cloudflare --port 50061 --invite team-alpha"
```

That prints a `distribai://join?...` link. Anyone can paste it:

```bash
python -m gridlink join "distribai://join?...&invite=team-alpha"     # print the plan
python -m gridlink join "distribai://join?...&invite=team-alpha" --exec   # start the worker
```

or open [`tools/gridlink/notebooks/colab_join.ipynb`](../tools/gridlink/notebooks/colab_join.ipynb)
and run the cells. Free GPU sessions, a spare laptop and a VPS all join the same
way: the worker only needs the URL and the invite code.

## Change how work is split

| Flag | Effect |
| --- | --- |
| `--replicas N` | one task per replica, each with its own seed |
| `--aggregate mean\|median\|trimmed_mean` | how replica gradients combine. `trimmed_mean` drops one high and one low per coordinate, so a single bad worker cannot drag the result |
| `--node-ttl S` | silence before a node counts as offline (default 60) |
| `--credit-divisor N` | training steps per credit (default 1000) |
| `--mem-mb`, `--cpu-sec`, `--fsize-mb` | rlimits the worker applies to the trainer child |

## Where things land

| Path | What |
| --- | --- |
| `build/cpp_port/` | binaries |
| `runtime/db/grid.db` | nodes, jobs, tasks, credits |
| `runtime/grid/jobs/<id>/` | bundle, result, aggregate envelope |
| `runtime/grid/tasks/<id>.env` | one envelope per replica |
| `runtime/grid/worker/<task>/trainer.log` | trainer stdout, on the worker machine |

Delete `runtime/grid` and `runtime/db/grid.db` for a clean slate. Both are
gitignored.

When a job finishes, the aggregate that produced it travels with the job: the
coordinator stores the `aggregate` name from the bundle, so `result.json` and the
job detail view always agree on how the replicas were combined.

## Check it works

Every gate below runs against the C++ port. Start with the fast ones.

```bash
make test            # parity, goldens, features
make grid-test       # coordinator plus workers on loopback, submits a real job
make grid-security-test   # only your own worker may report your own task
make grid-soak-test  # six workers and twelve jobs in one burst
make usage-test      # run, cancel, restart, resume, read the surfaces
make grid-all        # every grid and trainer gate
make suite-quick     # the test-suite subset, with sanitizers
```

A run is only green if the gate string says so. Skips count as skips, not
passes, and the summary line names what did not run.

`make screenshots` rebuilds `docs/assets/dashboard-*.png` from a live grid with
five workers and three jobs. It needs Chrome and is not part of the gates.

## When something goes wrong

| Symptom | Check |
| --- | --- |
| Worker registers but never gets work | `curl /v1/health`, then `/v1/tasks`. A job has to be submitted first, and its tasks must be `queued`. |
| Job stuck at `running` with tasks `assigned` | The worker stopped heartbeating. Wait out `--node-ttl`, or restart the worker. |
| `trainer not found at ...` on the worker | Build it there: `make torch`. Workers run the trainer locally. |
| `invite code required` | Pass the same `--invite` to the worker. |
| `task is assigned to another node` | A worker tried to report a replica it was not given. Check for two processes sharing a `--node-id`. |
| `submit without the token is refused` | Send `X-Grid-Token`, or run the submit command on the coordinator host. |
| Dashboard shows a missing-asset message | Start the coordinator with `--web-dir tools/cpp_port/grid/web`. |

## Next

- [Architecture](architecture.md) for the diagrams and the wire protocol.
- [Operations](operations.md) for exposing, capacity and drills.
- [Testing](testing.md) for the gates that cover your change, and for how to add
  one.
