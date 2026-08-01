# DistribAI System Architecture

The supported production layout is a single Python orchestrator process, one
Python worker daemon per machine, and Express/desktop UIs for operators and
contributors. Older Docker, Kubernetes, or Rust microservice diagrams are
research sketches only — they are not the path this repository ships.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     External clients                         │
│  (CLI, SDKs, third-party integrations)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 Admin HTTP / REST surface                    │
│  Port 8766 — jobs, monitoring, credits                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Single-process orchestrator                     │
│  services_python/orchestrator_grpc.py                        │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Job queue   │  │ Scheduler   │  │ Credit ledger       │ │
│  │             │  │ loop        │  │ (hash-chained)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Voting      │  │ Byzantine   │  │ Gradient            │ │
│  │             │  │ filters     │  │ compression         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ PoC         │  │ Sybil       │  │ Rate                │ │
│  │ challenges  │  │ heuristics  │  │ limiting            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ gRPC        │      │ SQLite      │      │ S3          │
│ streaming   │      │ (durable    │      │ (weights,   │
│ port 50051  │      │  state)     │      │  gradients) │
└─────────────┘      └─────────────┘      └─────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Worker fleet                           │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ gRPC daemon │  │ Training    │  │ Benchmark           │ │
│  │ client      │  │ executor    │  │ manager             │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ ML core     │  │ Local       │  │ Credit ledger       │ │
│  │ (families)  │  │ Byzantine   │  │ client              │ │
│  │             │  │ checks      │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### Monolithic Orchestrator

The full control plane runs inside one Python process.

**Entry:** `services_python/orchestrator_grpc.py`

#### Core Modules

1. **NodeService** — owns the gRPC session
   - Bidirectional streams to workers
   - Registration, heartbeats, progress, results
   - Hands tasks to idle nodes

2. **Job Queue Manager**
   - Jobs and tasks persisted in SQLite
   - Priority tiers (P0–P3)
   - Breaks work into micro-tasks
   - Requeues stale assignments

3. **Scheduler Loop**
   - Ticks roughly every 5 seconds
   - Pairs queued work with free capacity
   - Ages heartbeats (≈30s degraded, ≈50s offline)
   - Refreshes ETA and queue position

4. **Credit Ledger**
   - Hash-chained, append-oriented records
   - Merkle verification root
   - Reliability / early-adopter / surge multipliers
   - Peer-to-peer transfers

5. **Voting System**
   - Credit-weighted priority votes on jobs
   - Quorum rules with persisted vote rows
   - Sybil-aware checks on cast paths

6. **Byzantine Detection**
   - Multi-Krum-style outlier filters
   - Gradient norm screens
   - Trimmed-mean aggregation options
   - Statistical anomaly hooks

7. **Security Features**
   - Proof-of-Computation challenges
   - JWT sessions with expiry
   - Rate limits (~10 req/s class)
   - Pydantic / schema validation
   - Optional gRPC TLS

8. **Sybil Detection**
   - IP + hardware fingerprints
   - Vote-pattern heuristics
   - Reputation scoring

### Worker Daemon

**Entry:** `worker/src/daemon/daemon.py`

#### Components

1. **WorkerDaemon** — process supervisor
   - Keeps the gRPC stream alive
   - Reconnects with backoff
   - Heartbeats (~10s)
   - Coordinates task execution

2. **JobExecutor** — training runner
   - PyTorch (and sandboxed script) jobs
   - Pause/resume hooks
   - Progress throttled (~2s)
   - Gradient / artifact upload

3. **BenchmarkManager** — capability probe
   - GPU TFLOPS / bandwidth samples
   - Hardware inventory for scheduling

4. **ML Core** — first-party model builders
   - Architecture families (see `services_python/architecture_config.py`):
     `decoder_transformer`, `gru`, `gated_conv`, `moe_decoder`, `lstm`,
     `resnet_lm`, `hybrid_attn_rnn`, `dense_ffn`
   - Native size profiles (tiny → XL) plus an explicit custom profile builder
   - ToyModel gated behind `DISTRIBAI_ALLOW_TEST_MODELS=1`

5. **Byzantine Detector** — local quality signals
   - Norm checks and outlier hints before upload

### Data Storage

#### SQLite Database Schema

Authoritative DDL: [`runtime/db/schema.sql`](../../runtime/db/schema.sql). Conceptually:

```sql
-- Active nodes table
CREATE TABLE active_nodes (
    node_id TEXT PRIMARY KEY,
    status TEXT,              -- 'idle', 'working', 'degraded', 'offline'
    hardware_json TEXT,
    benchmark_json TEXT,
    reliability_score REAL,   -- 0.0 to 1.0
    jobs_completed INTEGER,
    jobs_failed INTEGER,
    contributing BOOLEAN,
    jwt_token TEXT,
    last_heartbeat INTEGER,
    created_at INTEGER
);

-- Jobs table
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    status TEXT,              -- 'queued', 'running', 'success', 'failed', 'cancelled'
    priority TEXT,            -- 'P0', 'P1', 'P2', 'P3'
    model_name TEXT,
    total_steps INTEGER,
    current_step INTEGER,
    progress_pct INTEGER,
    active_nodes INTEGER,
    queue_position INTEGER,
    estimated_start_hours REAL,
    hparams TEXT,
    created_at INTEGER,
    updated_at INTEGER
);

-- Tasks table (micro-tasks within jobs)
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    job_id TEXT REFERENCES jobs(job_id),
    node_id TEXT REFERENCES active_nodes(node_id),
    status TEXT,
    step_start INTEGER,
    step_end INTEGER,
    result_url TEXT,
    deadline_ts INTEGER,
    attempts INTEGER,
    created_at INTEGER
);

-- Credit ledger
CREATE TABLE credit_ledger (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT,
    amount REAL,
    balance_after REAL,
    tx_type TEXT,             -- 'job', 'vote', 'transfer', 'penalty'
    job_id TEXT,
    metadata TEXT,
    tx_hash TEXT,             -- Hash-chained
    prev_hash TEXT,
    created_at INTEGER
);

-- Vote transactions
CREATE TABLE vote_transactions (
    vote_id TEXT PRIMARY KEY,
    job_id TEXT,
    node_id TEXT,
    credits INTEGER,
    created_at INTEGER
);
```

#### Object storage (optional S3)

- Weights: `s3://<bucket>/weights/{job_id}/{version}.pt`
- Gradients: `s3://<bucket>/gradients/{job_id}/{task_id}/{round}.pt`
- Datasets: `s3://<bucket>/datasets/{dataset_id}/`
- Checkpoints: `s3://<bucket>/checkpoints/{job_id}/{step}.pt`

### Communication Protocol

#### gRPC Streaming (Primary)

```
┌─────────────┐                    ┌─────────────┐
│   Worker    │                    │ Orchestrator│
└─────────────┘                    └─────────────┘
      │                                  │
      │ 1. Connect                       │
      ├─────────────────────────────────>│
      │                                  │
      │ 2. RegisterSession               │
      │  {node_id, jwt, hardware}        │
      ├─────────────────────────────────>│
      │                                  │
      │ 3. RegisterAck                   │
      │  {session_token}                 │
      │<─────────────────────────────────│
      │                                  │
      │ 4. Heartbeat (every 10s)         │
      │  {seq, vram_free, gpu_util}     │
      ├─────────────────────────────────>│
      │                                  │
      │ 5. HeartbeatAck                  │
      │  {seq}                           │
      │<─────────────────────────────────│
      │                                  │
      │ 6. TaskAssign (when available)   │
      │  {task_id, job_id, model}       │
      │<─────────────────────────────────│
      │                                  │
      │ 7. TaskProgress (throttled)      │
      │  {step, loss}                    │
      ├─────────────────────────────────>│
      │                                  │
      │ 8. TaskResult                    │
      │  {status, gradient_url}          │
      ├─────────────────────────────────>│
```

#### REST API (Admin)

- HTTP/JSON on `ADMIN_PORT`
- Bearer JWT / admin secret
- Paginated list routes
- SSE at `/admin/stream` for operator live feeds

## Data Flow

### Job Lifecycle

```
1. Job Creation (REST API)
   ↓
2. Job Decomposition (Orchestrator)
   - Split into micro-tasks
   - Assign priority
   ↓
3. Queue Management
   - Add to priority queue
   - Calculate estimated start
   ↓
4. Task Assignment (Scheduler)
   - Match task to idle node
   - Send TaskAssign via gRPC
   ↓
5. Training Execution (Worker)
   - Load model weights from S3
   - Execute training steps
   - Report progress
   ↓
6. Result Collection
   - Upload gradients to S3
   - Send TaskResult via gRPC
   ↓
7. Aggregation (Orchestrator)
   - Collect all task results
   - Apply Byzantine detection
   - Aggregate gradients
   ↓
8. Credit Distribution
   - Calculate credits earned
   - Update ledger
   - Mark job complete
```

## Scaling Considerations

### Current Architecture (Monolithic)

**Pros:**
- Straightforward deploy story
- No cross-service hops on the hot path
- Easier local debugging
- Comfortable under roughly a thousand nodes

**Cons:**
- Single process failure domain
- Vertical scale ceiling
- Shared CPU/RAM among scheduler, admin, and aggregation

### Scaling Limits

| Metric | Current Limit | Bottleneck |
|--------|---------------|------------|
| Nodes | ~1000 | SQLite concurrency |
| Jobs/hour | ~500 | Scheduler loop frequency |
| Gradients/sec | ~100 | Network bandwidth |

### Horizontal Scaling Strategy

Beyond ~1000 nodes:

1. **Database sharding** — hash on `node_id`; stay on SQLite until metrics force a move
2. **Orchestrator sharding** — regional routing + cross-region aggregation
3. **Read replicas** — admin reads off replicas; writes stay primary

## Security Architecture

```
┌─────────────────────────────────────────┐
│         Defense layers                  │
├─────────────────────────────────────────┤
│ 1. Network                              │
│    - TLS for gRPC                       │
│    - JWT authentication                 │
│    - Rate limiting                      │
├─────────────────────────────────────────┤
│ 2. Application                          │
│    - Input validation                   │
│    - PoC challenges                     │
│    - Sybil detection                    │
├─────────────────────────────────────────┤
│ 3. Data                                 │
│    - Hash-chained ledger                │
│    - Signed credits                     │
│    - S3 with signed URLs                │
├─────────────────────────────────────────┤
│ 4. Execution                            │
│    - Byzantine fault tolerance          │
│    - Gradient validation                │
│    - Statistical outlier detection      │
└─────────────────────────────────────────┘
```

## Monitoring & Observability

### Metrics

| Metric | Type | Collection |
|--------|------|------------|
| Node heartbeats | Counter | gRPC |
| Job duration | Histogram | Scheduler |
| Credit transactions | Counter | Ledger |
| Gradient norms | Gauge | Byzantine detector |
| API latency | Histogram | Middleware |

### Health Checks

- `/admin/health` — orchestrator liveness
- Heartbeat timeouts — ~30s degraded / ~50s offline
- Job deadlines — automatic requeue

### Alerting

- Node offline > 60s
- Job failed > 3 attempts
- Byzantine nodes detected
- Credit system anomalies
