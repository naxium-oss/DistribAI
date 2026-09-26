# Architecture

What runs, what talks to what, and where the state lives. Every diagram below is
Mermaid, so it renders on GitHub and in most editors.

---

## The whole system

```mermaid
flowchart TB
  subgraph operator["Operator machine"]
    ORCH["distribai_orch<br/>coordinator"]
    DASH["Dashboard<br/>web/ assets"]
    LEDGER[("credit_ledger")]
    JOBS[["runtime/grid/jobs/<br/>bundles, results"]]
    TASKS[["runtime/grid/tasks/<br/>result envelopes"]]
    DB[("runtime/db/grid.db<br/>nodes, jobs, tasks, credits")]
  end

  subgraph workerA["Worker: local GPU"]
    WA["distribai_worker"]
    TA["distribai_torch_train"]
  end

  subgraph workerB["Worker: Colab / Kaggle / Molab / VPS"]
    WB["distribai_worker"]
    TB["distribai_torch_train"]
  end

  TRANSLATE["trainer_translate<br/>Python trainer to job dir"]
  TUNNEL["cloudflared quick tunnel<br/>free, no account"]

  TRANSLATE -->|job dir| ORCH
  ORCH --- DB
  ORCH --- JOBS
  ORCH --- TASKS
  ORCH --- LEDGER
  DASH -->|GET /v1/summary| ORCH
  WA <-->|HTTP: register, claim, result| ORCH
  WB <-->|HTTP over TLS through the tunnel| TUNNEL
  TUNNEL <-->|HTTP| ORCH
  WA -->|fork, rlimits| TA
  WB -->|fork, rlimits| TB
```

The coordinator is the only stateful piece. Workers hold no state between
replicas beyond their scratch directory.

---

## One job, end to end

```mermaid
sequenceDiagram
  autonumber
  participant Op as Operator
  participant Or as distribai_orch
  participant Db as SQLite
  participant W1 as worker 1
  participant W2 as worker 2
  participant Tr as trainer (libtorch)

  Op->>Or: POST /v1/jobs {job_dir, replicas: 2}
  Or->>Or: snapshot the bundle into jobs/<id>/bundle
  Or->>Db: INSERT jobs + 2 tasks
  Or-->>Op: 202 {job_id}

  W1->>Or: POST /v1/register {node_id, invite}
  Or->>Db: upsert active_nodes, rotate session token
  Or-->>W1: welcome {session_token, poll_s}

  W2->>Or: POST /v1/register
  Or-->>W2: welcome {session_token, poll_s}

  W1->>Or: POST /v1/claim
  Or->>Db: claim task r0 in one transaction
  Or-->>W1: task {spec with seed 42, files}
  W2->>Or: POST /v1/claim
  Or->>Db: claim task r1
  Or-->>W2: task {spec with seed 43, files}

  W1->>Tr: fork under RLIMIT_AS/CPU/FSIZE/NPROC/NOFILE/CORE
  Tr-->>W1: GradReport envelope + one JSON metrics line
  W1->>Or: POST /v1/result {envelope_b64}
  Or->>Db: task done, progress from steps
  Or->>Or: store tasks/<task_id>.env

  W2->>Tr: fork under rlimits
  Tr-->>W2: envelope
  W2->>Or: POST /v1/result

  Or->>Or: aggregate mean / median / trimmed_mean
  Or->>Or: write aggregate.env and result.json
  Or->>Db: job completed, credit_ledger rows per contributing node
  Or-->>W2: ack (job closed)
  Op->>Or: GET /v1/jobs/<id>
  Or-->>Op: per-replica metrics + aggregate
```

The `aggregate` method travels with the job: the coordinator reads it from the
job's own `job.json` and falls back to its `--aggregate` flag, then stores it on
the job row, so `GET /v1/jobs/<id>` and `result.json` always report the method
that produced the numbers.

The request that reports the last replica closes the job. It marks the job
`finalizing` in one conditional UPDATE, and only the winner writes the aggregate
and the credit rows, so a result arriving at the same moment as the maintenance
sweep cannot pay twice or write two different aggregates. The sweep still runs
every 20 seconds as a safety net for jobs nobody is waiting on any more, and it
returns a job stuck in `finalizing` for more than 30 seconds back to `running`.

---

## Failure handling

```mermaid
flowchart LR
  HB["worker heartbeat<br/>every 10s"] -->|missing for node-ttl| OFF["node marked offline"]
  OFF --> RE{"task attempts<br/>below max?"}
  RE -->|yes| QUEUE["task back to queued<br/>last_error: assignee went offline"]
  RE -->|no| FAIL["task failed"]
  QUEUE --> CLAIM["next worker claims it"]
  FAIL --> ALLFAIL{"any replica<br/>still alive?"}
  ALLFAIL -->|no| JOBFAIL["job failed with a reason"]
  JOBFAIL --> REPORT["result.json and dashboard show why"]
```

A worker that dies mid-task is treated the same way as a network partition:
silence past `--node-ttl` releases the replica. Tasks carry `attempt_count` and
`max_attempts` (3), so a job that kills every worker eventually stops instead of
looping.

A worker is also patient in the other direction. Registration retries for
`--register-wait` seconds (20 by default) when the coordinator cannot be reached
yet, which matters when the worker starts a moment before a tunnel or a
coordinator is ready. A refusal the coordinator actually sent, such as a bad
invite code, is final and is not retried. If a worker's session is rotated out
from under it, the next claim or result re-registers and continues, and a result
whose report fails is retried before the replica is abandoned.

---

## Persistence

```mermaid
erDiagram
  active_nodes ||--o{ tasks : "assignee"
  jobs ||--o{ tasks : "contains"
  jobs ||--o{ vote_transactions : "votes"
  active_nodes ||--o{ credit_ledger : "earns"
  active_nodes {
    text node_id PK
    text session_token
    text hardware_json
    text benchmark_json
    text status
    int last_heartbeat_ts
    int jobs_completed
    int jobs_failed
    real reliability_score
  }
  jobs {
    text job_id PK
    text model_name
    text dataset_ref
    text status
    int steps
    real progress_pct
    int current_step
    int total_steps
    text aggregate
  }
  tasks {
    text task_id PK
    text job_id FK
    text assignee_node_id
    text status
    text gradient_blob_url
    text output_json
    int steps
    int attempt_count
    int max_attempts
  }
  credit_ledger {
    int tx_id PK
    text node_id
    text tx_type
    real amount
    real balance_after
    text tx_hash
    text prev_hash
  }
```

Tables come from [`runtime/db/schema.sql`](../runtime/db/schema.sql), which the
coordinator applies at startup. `CREATE TABLE IF NOT EXISTS` leaves an existing
database alone, so columns added later are also applied as a small migration by
hand (currently `jobs.aggregate`); a grid keeps working across an upgrade without
recreating its database. Large or binary values stay out of the database: job
bundles, per-task envelopes and results are files, and the rows hold paths and
metrics.

---

## Trust boundaries

```mermaid
flowchart TB
  subgraph public["Public side"]
    TUN["tunnel URL (https, port 443)"]
    ANY["anyone with the URL"]
  end
  subgraph local["Loopback / LAN"]
    ORCH2["coordinator"]
    ADM["operator writes<br/>POST /v1/jobs, /cancel"]
  end
  ANY --> TUN --> ORCH2
  ORCH2 -.->|invite code gate at register| ANY
  ORCH2 -.->|X-Grid-Token gate| ADM
  ADM --> ORCH2
```

What the design assumes:

- Read endpoints are open. They show node ids, job names, losses and hashes,
  nothing that would let a stranger run work.
- Writes need `X-Grid-Token`. Without a configured token only loopback callers
  may submit or cancel, so an accidental public bind does not accept jobs.
- Worker registration needs the invite code when the coordinator was started
  with `--invite`. Tunnel URLs are public, so set one.
- A session token is bound to the node that registered and is rotated on every
  re-registration, so an old token stops working. A worker may only report on the
  task it was assigned: a result for another node's task is refused, which stops
  a registered stranger from failing or finishing someone else's replica.
- An operator cancel fails the queued and assigned tasks and moves the job to
  `cancelled`. A worker that finishes training afterwards is answered with an
  acknowledgement, but finalization only claims a job that is still `queued` or
  `running`, so a cancelled job never reopens and never writes a result.
- The trainer child inherits no ambient environment beyond `PATH` and `TMPDIR`,
  and runs under rlimits plus its own PID, user and mount namespaces where the
  kernel allows them. See [`native sandbox`](quickstart.md).
- The dashboard never renders worker-supplied strings as HTML: everything goes
  through `textContent`.

---

## Where code lives

```mermaid
flowchart LR
  subgraph port["tools/cpp_port (primary)"]
    CORE["core/: tensor, parity, envelope, checkpoint, multi_model"]
    SAND["sandbox/: rlimits + namespaces"]
    TORCH["torch/: LibTorch runner + job spec"]
    GRID["grid/: coordinator, worker, store, HTTP, dashboard"]
    ABIP["abi/, gpu/, apps/"]
    GATES["tests/: goldens, suite, parity, grid e2e"]
  end
  subgraph py["tools/ (Python, thin)"]
    TT["trainer_translate/"]
    GL["gridlink/"]
    BENCH["bench/run_limited.sh"]
  end
  subgraph frozen["legacy/ (deprecated)"]
    LEG["Python orchestrator, worker, Node dashboards"]
  end
  CORE --> TORCH --> GRID
  SAND --> GRID
  TT -->|job dir| GRID
  GL -->|expose + join| GRID
  GATES --> GRID
  LEG -.->|same schema, old control plane| CORE
```

The Python tools that remain are the ones where a Python dependency is the point:
`trainer_translate` needs torch to trace a user's trainer, and `gridlink` is a
thirteen-file operator utility. Everything on the training path is C++.

---

## Read next

- [Quickstart](quickstart.md): build and run in a few commands.
- [Operations](operations.md): exposing the grid, capacity, failure drills.
- [Testing](testing.md): the gates and what each one proves.
- [From Python to C++](from-python-to-cpp.md): the deprecation and the mapping.
