# DistribAI API Reference

HTTP admin routes and gRPC session contracts for the live orchestrator. Prefer this file over README summaries when wiring clients.

---

## Base URLs

| Environment | gRPC | REST Admin |
|-------------|------|------------|
| Local | `localhost:50051` | `http://localhost:8766` |
| Remote | `<your-grpc-host>:50051` | `https://<your-admin-host>:8766` |

Local defaults follow `GRPC_PORT` / `ADMIN_PORT`. Remote hostnames are whatever your operator publishes — there is no shared public SaaS endpoint.

---

## Authentication

Pass a Bearer token on protected routes:

```bash
curl -H "Authorization: Bearer <jwt_token>" http://localhost:8766/admin/jobs
```

Token kinds in use today:

- `node` — worker daemon sessions
- `admin` — operator / admin API (often `DISTRIBAI_ADMIN_SECRET`)
- `user` — dashboard and voting surfaces where issued

Health probes may stay open; job and node mutations require auth when lockdown is enabled (`ADMIN_REQUIRE_AUTH=1` or non-loopback `ADMIN_HOST`).

Admin middleware also propagates **correlation IDs**: send `X-Request-Id` (or `X-Correlation-Id`); the same value is echoed on the response and included in structured log records as `correlation_id`.

---

## REST Endpoints

### Health & Status

#### GET /admin/health
Health check endpoint.

**Response:**
```json
{
  "ok": true,
  "timestamp": 1714396800,
  "version": "0.1.0",
  "active_nodes": 42,
  "queued_jobs": 5,
  "running_jobs": 3,
  "job_submission_available": true
}
```

#### GET /admin/stats
Aggregated operator dashboard counters (connected nodes, running jobs, credits, TFLOPS).

#### GET /api/operator/status
Operator truth banner payload (TLS flags, ledger parity, secret-from-env hints).

---

### Jobs

#### POST /admin/jobs
Create a new training job.

**Request:**
```json
{
  "job_type": "fine_tune",
  "base_model": "DistribAI/DistribAI-LM-Small-Base",
  "dataset_ref": "s3://distribai-datasets/small-v2.jsonl",
  "hyperparams": {
    "lr": 1e-4,
    "epochs": 3,
    "batch_size": 32,
    "gradient_accumulation_steps": 4
  },
  "steps": 100,
  "batch_size": 32,
  "script_package_b64": "<optional base64 tar.gz with run.py; max ~5MB decoded>",
  "priority_tier": "P1",
  "checkpoint_interval": 25,
  "gradient_compression": {
    "enabled": true,
    "method": "powersgd",
    "rank": 8
  },
  "architecture_config": {
    "version": 1,
    "family": "moe_decoder",
    "dim": 256,
    "ffn_dim": 1024,
    "n_logical_layers": 8,
    "num_experts": 4,
    "top_k": 2,
    "seq_len": 2048
  },
  "hparams": {
    "callback_url": "https://example.com/hooks/job-complete"
  }
}
```

`architecture_config` is a pure-data JSON definition. It is validated before the job is queued, persisted inside each task's `hparams_json`, and reconstructed by workers from a registered family—no Python imports or uploaded executable model code are accepted. Supported families are exactly `decoder_transformer`, `gru`, `gated_conv`, `moe_decoder`, `lstm`, `resnet_lm`, `hybrid_attn_rnn`, and `dense_ffn`; `family` and the legacy `architecture` alias must agree when both are supplied. The configuration is bounded to 64 KiB, nesting depth 6, and version `1`. Integer bounds are: `dim` 16–4096, unique layers 1–64, logical layers 1–128, attention heads 1–64, feed-forward width 16–16384, context length 8–32768 (transformers and hybrid attention capped at 8192), experts 1–64, top-k 1–16, convolution kernel 2–31, and GRU/LSTM layers 1–16 (`gru_layers`); dropout is limited to 0–0.5. A derived native parameter estimate is capped at 512 million parameters. Existing named profiles such as `distribai-small`, `distribai-lstm-small`, `distribai-resnet-tiny`, `distribai-hybrid-small`, and `distribai-dense-tiny` remain supported for API compatibility, but new clients should submit an architecture definition.

For a file-based workflow, upload the contents of an `.json` file as the `architecture_config` object; the dashboard's **Create Job** form provides both a JSON editor and local file picker. The same object is accepted by `POST /v1/jobs`.

Optional keys inside `hparams` / `hyperparams` (persisted as JSON on the job row):

| Key | Type | Notes |
|-----|------|-------|
| `callback_url` | string | On job terminal states (`success` / `failed` / `cancelled` / `timeout` / `error`), DistribAI POSTs a signed JSON payload to this URL. Header `X-DistribAI-Signature: sha256=<hex>` is HMAC-SHA256 of the raw body using `SIGNING_KEY`. Loopback URLs are rejected unless `DISTRIBAI_ALLOW_LOOPBACK_WEBHOOKS=1`. |
| `package_sha256` | string | Expected bundle digest when submitting script packages |
| `execution_paradigm` | string | Set to `script` when `script_package_b64` is present |

**Priority lanes:** set top-level `priority_tier` to `P0` (highest) … `P3` (lowest). The scheduler orders queued tasks by lane, then numeric `priority`, votes, and age. List jobs with `?priority_tier=P0` or `?lane=P0,P1`.

**Correlation IDs:** every admin HTTP response includes `X-Request-Id` / `X-Correlation-Id`. Clients may send either header to propagate an existing id; otherwise the orchestrator mints one and attaches it to structured logs as `correlation_id`.

**Response:**
```json
{
  "ok": true,
  "job_id": "job-20240429-abc123",
  "task_id": "task-def456",
  "status": "queued",
  "estimated_start": "2024-04-29T10:30:00Z",
  "queue_position": 3
}
```

**Errors:**
- `400`: Invalid request parameters
- `401`: Authentication required
- `429`: Rate limit exceeded

#### Script package format (gRPC `TaskAssign.script_package`)

When job submission distributes custom code, workers receive a **gzip tar** (`application/gzip` tarball bytes) with:

| Path | Required | Purpose |
|------|----------|---------|
| `run.py` | yes | Entry script executed in an isolated task directory |
| `requirements.txt` | no | Installed with `pip install --target <task>/.site-packages` |
| `config.json` | no | Metadata (`job_id`, `job_type`, …) exposed via env |
| `hyperparams.json` | written by worker | Scheduler/orchestrator hyperparameters |

Tar members with path traversal (`../`), symlinks, or device nodes are **rejected**. See `worker/src/daemon/script_runner.py`.

Poll job status: `GET /admin/jobs/{job_id}` or `GET /v1/jobs/{job_id}` (Bearer JWT for v1).

**Bundle persistence:** optional `script_package_b64` on `POST /admin/jobs` is stored under `runtime/bundles/{task_id}.tar.gz` (override with `DISTRIBAI_BUNDLE_DIR`). The scheduler loads from memory or disk before gRPC assign.

#### Artifact egress (gradients, checkpoints, script bundles)

| Artifact | When `S3_*` configured | When S3 omitted (local dev) |
|----------|----------------------|-----------------------------|
| Training gradients | Presigned `s3://{S3_BUCKET_NAME}/gradients/...` URLs; worker uploads via allowlisted hosts (`blob_url_policy.py`) | `file://` or local paths under orchestrator/worker temp dirs; uploads may be skipped with warnings |
| Script bundles (admin create) | Also uploaded to `s3://{bucket}/bundles/{task_id}.tar.gz` when `S3_BUCKET_NAME` is set | On-disk `runtime/bundles/` (scheduler loads disk then S3) |
| Job status / task output | SQLite `tasks.output_json`, `jobs.latest_reason` | Same |

Production multi-instance orchestrators should plan shared storage (S3 or NFS) for bundles and gradients; a single SQLite file must have one writer (`docs/runbooks/deployment.md`).

---

#### GET /admin/jobs
List all jobs with pagination.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| page | int | 1 | Page number |
| per_page | int | 20 | Items per page (max 100) |
| sort_by | string | created_ts | Sort field |
| sort_order | string | desc | asc or desc |
| status | string | - | Filter by status |
| priority_tier / lane | string | - | Filter by priority lane (`P0`–`P3`; comma-separated allowed) |

**Query parameters (implemented):** `active_only` (default `true`), `include_history` (default `false`), `priority_tier` / `lane`.

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job-20240429-abc123",
      "status": "queued",
      "job_type": "fine_tune",
      "base_model": "DistribAI/DistribAI-LM-Small-Base",
      "progress_pct": 0,
      "queue_blockers": [
        {
          "code": "no_workers_connected",
          "message": "No workers are connected on gRPC; tasks cannot be assigned."
        }
      ]
    }
  ],
  "queue_depth": 3,
  "queue_fleet": {
    "queue_depth": 3,
    "connected_count": 0,
    "idle_count": 0,
    "busy_count": 0,
    "offline_registered_count": 2
  },
  "pagination": {
    "total": 156,
    "page": 1,
    "per_page": 20,
    "total_pages": 8
  },
  "queue_depth": 12
}
```

**Headers:**
```
X-Total-Count: 156
X-Page: 1
X-Per-Page: 20
X-Total-Pages: 8
```

---

#### GET /admin/jobs/{job_id}
Get detailed job information.

**Response:**
```json
{
  "job_id": "job-20240429-abc123",
  "status": "running",
  "job_type": "fine_tune",
  "base_model": "DistribAI/DistribAI-LM-Small-Base",
  "dataset_ref": "s3://distribai-datasets/small-v2.jsonl",
  "hyperparams": {
    "lr": 1e-4,
    "batch_size": 32
  },
  "progress": {
    "current_step": 45,
    "total_steps": 100,
    "pct": 45.5,
    "current_loss": 2.34,
    "best_loss": 1.87
  },
  "nodes": [
    {
      "node_id": "worker-rtx4090-01",
      "status": "working",
      "assigned_task": "task-def456",
      "last_heartbeat": 1714396800
    }
  ],
  "checkpoints": [
    {
      "step": 25,
      "url": "s3://bucket/checkpoints/job-abc123/step-25.pt",
      "timestamp": 1714396200,
      "loss": 2.45
    }
  ],
  "votes": {
    "total": 156,
    "credits_spent": 2340,
    "priority_score": 2456.5
  },
  "created_ts": 1714395600,
  "started_ts": 1714396000,
  "estimated_completion": "2024-04-29T12:00:00Z"
}
```

---

#### GET /admin/jobs/compare?a={job_id}&b={job_id}
Side-by-side summary for two jobs (status, steps, failure_code).

#### GET /admin/jobs/{job_id}/artifacts
List on-disk script bundle and gradient checkpoint paths for the job's latest task.

#### POST /admin/jobs/{job_id}/retry
Re-queue a terminal job (`failed`, `cancelled`, `success`, etc.) for operator retry.

#### DELETE /admin/jobs/{job_id}
Cancel a queued or running job.

**Response:**
```json
{
  "ok": true,
  "job_id": "job-20240429-abc123",
  "status": "cancelled",
  "affected_tasks": 3,
  "refunded_votes": 156
}
```

---

### Nodes

#### GET /admin/nodes
List all nodes with pagination.

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| status | string | - | Filter: online, offline, degraded |
| sort_by | string | last_heartbeat_ts | Sort field |
| sort_order | string | desc | asc or desc |

**Response:**
```json
{
  "nodes": [
    {
      "node_id": "worker-rtx4090-01",
      "status": "online",
      "hardware": {
        "gpu": "NVIDIA RTX 4090",
        "vram_gb": 24,
        "compute_score": 9876.5
      },
      "reliability": {
        "score": 0.98,
        "jobs_completed": 156,
        "jobs_failed": 2
      },
      "current_task": {
        "task_id": "task-def456",
        "job_id": "job-abc123",
        "progress_pct": 45
      },
      "credits": {
        "balance": 1540.5,
        "lifetime_earned": 2500.0
      },
      "last_heartbeat": 1714396800,
      "connected_since": 1714300000
    }
  ],
  "pagination": {
    "total": 42,
    "page": 1,
    "per_page": 20
  },
  "summary": {
    "online": 42,
    "offline": 3,
    "degraded": 1,
    "total_compute_score": 356789.0
  }
}
```

---

#### GET /admin/nodes/{node_id}
Get detailed node information.

**Response:**
```json
{
  "node_id": "worker-rtx4090-01",
  "status": "online",
  "session_token": "sess-***",
  "hardware": {
    "platform": "Linux-5.15-x86_64",
    "cpu": "AMD Ryzen 9 7950X",
    "cpu_cores": 16,
    "ram_gb": 64,
    "gpu": "NVIDIA RTX 4090",
    "vram_gb": 24,
    "cuda_version": "12.2"
  },
  "benchmark": {
    "overall_score": 9876.5,
    "tensor_score": 11200.0,
    "vram_score": 2400.0,
    "network_score": 850.0,
    "tier": "S",
    "last_benchmark": 1714300000
  },
  "performance": {
    "jobs_completed": 156,
    "jobs_failed": 2,
    "avg_task_time_sec": 342.5,
    "total_compute_hours": 452.3
  },
  "reliability": {
    "score": 0.98,
    "uptime_pct": 99.2,
    "consecutive_success": 45
  },
  "byzantine": {
    "flagged_count": 0,
    "last_flagged": null,
    "reputation": 0.95
  },
  "credits": {
    "balance": 1540.5,
    "lifetime_earned": 2500.0,
    "lifetime_spent": 959.5,
    "pending": 0.0
  },
  "multipliers": {
    "surge_opt_in": true,
    "reliability_multiplier": 1.2,
    "current_rate": 1.5
  },
  "current_task": {
    "task_id": "task-def456",
    "job_id": "job-abc123",
    "assigned_at": 1714396400,
    "progress_pct": 45,
    "estimated_completion": 1714397200
  },
  "history": {
    "last_10_tasks": [...],
    "heartbeat_history_24h": [...]
  }
}
```

---

#### POST /admin/nodes/{node_id}/contributing
Set node contributing status.

**Request:**
```json
{
  "contributing": false,
  "reason": "maintenance",
  "duration_hours": 2
}
```

**Response:**
```json
{
  "ok": true,
  "node_id": "worker-rtx4090-01",
  "contributing": false,
  "previous_status": "online",
  "current_task": {
    "task_id": "task-def456",
    "graceful_exit": true,
    "checkpoint_saved": "s3://bucket/checkpoints/task-def456-emergency.pt"
  }
}
```

---

### Credits

#### GET /v1/credits/balance
Get credit balance (node endpoint).

**Response:**
```json
{
  "node_id": "worker-rtx4090-01",
  "confirmed": 1540.5,
  "pending": 0.0,
  "lifetime_earned": 2500.0,
  "lifetime_votes_cast": 156,
  "estimated_apr": 12.5
}
```

---

#### POST /v1/credits/transfer
Transfer credits to another node.

**Request:**
```json
{
  "recipient": "worker-rtx4090-02",
  "amount": 100.0,
  "memo": "Thanks for the help!"
}
```

**Response:**
```json
{
  "ok": true,
  "transfer_id": "xfer-abc123",
  "sender": "worker-rtx4090-01",
  "recipient": "worker-rtx4090-02",
  "amount": 100.0,
  "sender_balance": 1440.5,
  "recipient_balance": 890.0,
  "timestamp": 1714396800
}
```

---

#### GET /admin/credits
List all credit balances (admin).

**Response:**
```json
{
  "accounts": [
    {
      "node_id": "worker-rtx4090-01",
      "balance": 1540.5,
      "lifetime_earned": 2500.0,
      "lifetime_spent": 959.5,
      "votes_cast": 156,
      "rank": 3
    }
  ],
  "summary": {
    "total_circulating": 125000.0,
    "total_accounts": 156,
    "avg_balance": 801.28,
    "median_balance": 450.0
  }
}
```

---

### Voting

#### POST /v1/votes
Cast a vote on a job.

**Request:**
```json
{
  "job_id": "job-20240429-abc123",
  "credits_to_spend": 50,
  "memo": "Important research!"
}
```

**Response:**
```json
{
  "ok": true,
  "vote_id": "vote-def789",
  "job_id": "job-20240429-abc123",
  "credits_spent": 50,
  "new_priority_score": 2456.5,
  "queue_position_change": -2,
  "new_balance": 1490.5,
  "receipt": {
    "signature": "sig-...",
    "timestamp": 1714396800,
    "tx_hash": "0x..."
  }
}
```

---

#### GET /v1/votes/active
List active votes and proposals.

**Response:**
```json
{
  "votes": [
    {
      "id": "vote-gov-001",
      "type": "governance",
      "title": "Increase checkpoint interval",
      "description": "Proposal to increase default checkpoint interval from 25 to 50 steps",
      "proposer": "worker-rtx4090-01",
      "options": ["yes", "no", "abstain"],
      "status": "pending",
      "created_at": 1714390000,
      "expires_at": 1714476400,
      "quorum_required": 0.7,
      "participation_pct": 0.45,
      "results": {
        "yes": 2340,
        "no": 890,
        "abstain": 150
      }
    }
  ],
  "my_votes": {
    "vote-gov-001": "yes"
  }
}
```

---

### Byzantine Detection

#### GET /admin/byzantine-stats
Get Byzantine detection statistics.

**Response:**
```json
{
  "current_method": "multi_krum",
  "flagged_nodes": [
    {
      "node_id": "worker-suspicious-01",
      "anomaly_score": 0.95,
      "detection_method": "clustering",
      "flagged_at": 1714396800,
      "gradient_history": [...],
      "reputation": 0.15,
      "action_taken": "quarantined"
    }
  ],
  "stats": {
    "total_gradients_processed": 125000,
    "anomalies_detected": 45,
    "false_positives": 2,
    "true_positives": 43,
    "current_threshold": 0.75
  },
  "recent_detections": [...]
}
```

---

#### GET /admin/nodes/{node_id}/sybil-report
Get Sybil detection report for a node.

**Response:**
```json
{
  "node_id": "worker-rtx4090-01",
  "risk_score": 0.12,
  "risk_level": "low",
  "indicators": [
    {
      "type": "account_age",
      "value": "45 days",
      "risk_contribution": 0.0
    },
    {
      "type": "hardware_fingerprint",
      "value": "unique",
      "risk_contribution": 0.0
    }
  ],
  "related_accounts": [],
  "recommendation": "No action needed"
}
```

---

## gRPC Endpoints

### NodeService

**Proto:** `distribai.proto`

#### stream_session
Bidirectional streaming for worker node communication.

**Client → Server Messages:**

1. **Register**
```protobuf
message ClientMessage {
  oneof payload {
    RegisterRequest register = 1;
    Heartbeat heartbeat = 2;
    TaskResult result = 3;
    ProgressUpdate progress = 4;
    LogMessage log = 5;
  }
}
```

2. **Heartbeat**
```protobuf
message Heartbeat {
  string node_id = 1;
  int32 seq = 2;
  float gpu_util = 3;
  float vram_used_gb = 4;
  string current_task = 5;
  int32 current_step = 6;
}
```

**Server → Client Messages:**

1. **RegisterAck**
```protobuf
message ServerMessage {
  oneof payload {
    RegisterAck register_ack = 1;
    HeartbeatAck heartbeat_ack = 2;
    TaskAssign task_assign = 3;
    ControlCommand control = 4;
  }
}
```

2. **TaskAssign**
```protobuf
message TaskAssign {
  string task_id = 1;
  string job_id = 2;
  string job_type = 3;
  string base_model = 4;
  string dataset_ref = 5;
  int32 steps = 6;
  int32 step_offset = 7;
  string weight_url = 8;
  map<string, string> hyperparams = 9;
  int32 deadline_ts = 10;
}
```

---

### Example Client Code

**Python:**
```python
import grpc
from worker.src.distribai_proto import distribai_pb2, distribai_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = distribai_pb2_grpc.NodeServiceStub(channel)

def message_generator():
    # Send register
    yield distribai_pb2.ClientMessage(
        register=distribai_pb2.RegisterRequest(
            node_id="worker-001",
            hardware_json='{"gpu": "RTX 4090"}'
        )
    )
    
    # Send heartbeats
    for seq in range(1000):
        yield distribai_pb2.ClientMessage(
            heartbeat=distribai_pb2.Heartbeat(
                node_id="worker-001",
                seq=seq,
                gpu_util=0.85
            )
        )
        time.sleep(10)

# Start streaming
responses = stub.stream_session(message_generator())
for response in responses:
    if response.HasField('task_assign'):
        print(f"Assigned task: {response.task_assign.task_id}")
```

---

## Rate Limits

| Endpoint Type | Rate Limit | Burst |
|---------------|------------|-------|
| Public read | 10 req/s | 20 |
| Public write | 5 req/s | 10 |
| Node endpoints | 100 req/s | 200 |
| Admin endpoints | 50 req/s | 100 |

**Rate Limit Headers:**
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1714396860
```

---

## Error Codes

| Code | Meaning | Recovery |
|------|---------|----------|
| 400 | Bad Request | Check request format |
| 401 | Unauthorized | Refresh JWT token |
| 403 | Forbidden | Check permissions |
| 404 | Not Found | Verify resource exists |
| 429 | Rate Limited | Wait and retry |
| 500 | Server Error | Contact support |
| 503 | Service Unavailable | Retry with backoff |

**Error Response Format:**
```json
{
  "error": "invalid_parameter",
  "message": "batch_size must be between 1 and 512",
  "field": "hyperparams.batch_size",
  "request_id": "req-abc123"
}
```

---

## SDK Examples

### JavaScript/TypeScript

```typescript
import { DistribAIClient } from '@distribai/sdk';

const client = new DistribAIClient({
  baseURL: process.env.ORCHESTRATOR_ADMIN_URL || 'http://127.0.0.1:8766',
  token: process.env.DISTRIBAI_TOKEN
});

// List jobs
const jobs = await client.jobs.list({ status: 'running' });

// Create job
const job = await client.jobs.create({
  job_type: 'fine_tune',
  base_model: 'DistribAI/DistribAI-LM-Small-Base',
  steps: 100
});

// Cast vote
await client.votes.cast({
  job_id: job.id,
  credits: 50
});
```

### Python

```python
from distribai import Client

client = Client(token="your-jwt-token")

# Get node status
node = client.nodes.get("worker-rtx4090-01")
print(f"Status: {node.status}, Credits: {node.credits.balance}")

# Transfer credits
client.credits.transfer(
    recipient="worker-rtx4090-02",
    amount=100.0,
    memo="Thanks!"
)

# Stream logs
for log in client.jobs.stream_logs("job-abc123"):
    print(f"[{log.level}] {log.message}")
```

---

**API Version:** 1.0.0  
**Last Updated:** 2026-04-29
