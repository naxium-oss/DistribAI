# DistribAI API Documentation

> **Docs index:** [../README.md](../README.md). **Orchestrator entrypoint:** [`services_python/orchestrator_grpc.py`](../../services_python/orchestrator_grpc.py).

## Overview

DistribAI publishes two primary surfaces:
1. **gRPC Streaming API** — worker nodes (bidirectional stream)
2. **REST Admin API** — job control and monitoring

Model architecture families available for jobs (see `services_python/architecture_config.py`): `decoder_transformer`, `gru`, `gated_conv`, `moe_decoder`, `lstm`, `resnet_lm`, `hybrid_attn_rnn`, `dense_ffn`.

## Base URLs

- **gRPC**: `grpc://<orchestrator-host>:50051`
- **REST Admin**: `http://<orchestrator-host>:8766`

## Authentication

Two credential types apply:

| Surface | Credential | Header / field |
|---------|------------|----------------|
| **gRPC worker stream** | Node JWT from registration | `RegisterSession.jwt_token` on first message |
| **REST admin API** (`/admin/*`, proxied `/api/worker/*`) | `DISTRIBAI_ADMIN_SECRET` | `Authorization: Bearer <DISTRIBAI_ADMIN_SECRET>` |

Admin routes do **not** accept the node JWT as admin auth. When `ADMIN_REQUIRE_AUTH=1` or admin binds outside loopback, missing `DISTRIBAI_ADMIN_SECRET` yields **503**; wrong/missing Bearer yields **401**. See [`services_python/admin_auth.py`](../../services_python/admin_auth.py) and [deployment runbook](../runbooks/deployment.md).

Public registration and health endpoints stay unauthenticated unless registration PoC policy applies (see `REGISTRATION_REQUIRE_POC` in `.env.example`).

## gRPC API

> **Dedicated stub:** [grpc.md](./grpc.md) — proto path, stub regen, TLS pointers.

### Service: NodeService

```protobuf
service NodeService {
  rpc StreamSession (stream ClientMessage) returns (stream ServerMessage);
}
```

### Client Messages (Worker → Orchestrator)

#### RegisterSession
First message on connect — registers the node.

```protobuf
message RegisterSession {
  string node_id = 1;
  string jwt_token = 2;
  string hardware_json = 3;   // GPU specs, RAM, etc.
  string benchmark_json = 4;  // Benchmark results
  uint64 ts = 5;
}
```

#### Heartbeat
Emitted every 10 seconds to keep the session alive.

```protobuf
message Heartbeat {
  string node_id = 1;
  uint32 seq = 2;
  uint32 vram_free_mb = 3;
  float gpu_util = 4;
  optional string task_id = 5;
  uint64 ts = 6;
}
```

#### TaskProgress
Reports training progress (throttled to at most once per 2 seconds).

```protobuf
message TaskProgress {
  string node_id = 1;
  string job_id = 2;
  string task_id = 3;
  uint32 step = 4;
  float loss = 5;
  uint64 ts = 6;
}
```

#### TaskResult
Reports task completion or failure.

```protobuf
message TaskResult {
  string node_id = 1;
  string job_id = 2;
  string task_id = 3;
  string status = 4;              // "success", "failed", "rejected"
  string gradient_blob_url = 5;   // S3 URL
  uint32 wall_ms = 6;
  string reason = 7;
  string output_json = 8;
  uint64 ts = 9;
}
```

#### LogMessage
Forwards log lines to the orchestrator.

```protobuf
message LogMessage {
  string node_id = 1;
  string level = 2;    // "debug", "info", "warning", "error"
  string message = 3;
  uint64 ts = 4;
}
```

### Server Messages (Orchestrator → Worker)

#### RegisterAck
Returned after successful registration.

```protobuf
message RegisterAck {
  string session_token = 1;
  string server_version = 2;
}
```

#### HeartbeatAck
Acknowledges a heartbeat.

```protobuf
message HeartbeatAck {
  uint32 seq = 1;
}
```

#### TaskAssign
Dispatches a training task to the worker.

```protobuf
message TaskAssign {
  string task_id = 1;
  string job_id = 2;
  string model_name = 3;
  string weight_blob_url = 4;
  string batch_blob_url = 5;
  string hparams_json = 6;
  uint64 deadline_ts = 7;
  string weight_version = 8;
  int32 steps = 9;
}
```

#### ControlMessage
Operator / control-plane commands.

```protobuf
message ControlMessage {
  string action = 1;        // "pause", "resume", "drain", "cancel_job", "benchmark"
  optional string target_id = 2;
}
```

## REST Admin API

### Health

#### GET /admin/health
Liveness probe.

**Response:**
```json
{
  "ok": true,
  "timestamp": 1234567890,
  "version": "1.0.0"
}
```

### Jobs

#### GET /admin/jobs
List jobs with pagination.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `per_page` (int): Items per page (default: 20, max: 100)
- `status` (string): Filter by status

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "job-abc123",
      "status": "running",
      "progress_pct": 45,
      "current_step": 45,
      "total_steps": 100,
      "active_nodes": 3,
      "eta_seconds": 1200
    }
  ],
  "queue_depth": 5,
  "pagination": {
    "total": 50,
    "page": 1,
    "per_page": 20,
    "total_pages": 3
  }
}
```

#### GET /admin/jobs/{job_id}
Fetch a single job.

**Response:**
```json
{
  "job_id": "job-abc123",
  "status": "running",
  "progress_pct": 45,
  "current_step": 45,
  "total_steps": 100,
  "priority": "P0",
  "model_name": "distribai-small",
  "dataset_ref": "s3://bucket/dataset.pt",
  "hparams": {"lr": 0.001, "batch_size": 32},
  "created_at": 1234567890,
  "updated_at": 1234567900
}
```

#### POST /admin/jobs
Enqueue a new job.

**Request Body:**
```json
{
  "steps": 100,
  "batch_size": 32,
  "priority": "P1",
  "model_name": "distribai-small",
  "dataset_ref": "s3://bucket/dataset.pt",
  "hparams": {"lr": 0.001, "weight_decay": 0.0001},
  "steps_per_task": 25
}
```

**Response:**
```json
{
  "ok": true,
  "job_id": "job-abc123",
  "task_id": "task-xyz789",
  "status": "queued"
}
```

#### DELETE /admin/jobs/{job_id}
Cancel a job.

**Response:**
```json
{
  "ok": true,
  "job_id": "job-abc123",
  "status": "cancelled"
}
```

### Nodes

#### GET /admin/nodes
List all nodes.

**Response:**
```json
{
  "nodes": [
    {
      "node_id": "node-123",
      "status": "working",
      "hardware": {"gpu": "RTX 4090", "vram_mb": 24576},
      "benchmark": {"score": 15000},
      "reliability_score": 0.95,
      "jobs_completed": 10,
      "jobs_failed": 0,
      "last_seen": 1234567890
    }
  ],
  "total": 50
}
```

#### GET /admin/nodes/paginated
List nodes with pagination and sorting.

**Query Parameters:**
- `page` (int): Page number
- `per_page` (int): Items per page
- `sort_by` (string): Field to sort by
- `sort_order` (string): "asc" or "desc"

#### POST /admin/nodes/{node_id}/contributing
Set node contributing status.

**Request Body:**
```json
{
  "contributing": true
}
```

### Logs

#### GET /admin/logs
Read orchestrator logs.

**Query Parameters:**
- `lines` (int): Number of lines (default: 100, max: 1000)
- `level` (string): Filter by level ("debug", "info", "warning", "error")

**Response:**
```json
{
  "logs": [
    {
      "timestamp": 1234567890,
      "level": "info",
      "message": "Job assigned to node-123"
    }
  ]
}
```

### Credits

#### GET /admin/credits
List credit balances (paginated).

**Response:**
```json
{
  "credits": [
    {
      "node_id": "node-123",
      "balance": 1500,
      "lifetime": 2000,
      "votes_cast": 5
    }
  ],
  "pagination": {
    "total": 100,
    "page": 1,
    "per_page": 20
  }
}
```

#### GET /admin/credits/{node_id}
Credits for one node.

**Response:**
```json
{
  "node_id": "node-123",
  "balance": 1500,
  "lifetime": 2000,
  "votes_cast": 5,
  "transactions": [...]
}
```

### Votes

#### GET /admin/votes
List votes.

**Query Parameters:**
- `job_id` (string): Filter by job

**Response:**
```json
{
  "votes": [
    {
      "vote_id": "vote-abc",
      "job_id": "job-123",
      "node_id": "node-456",
      "credits": 100,
      "timestamp": 1234567890
    }
  ]
}
```

#### GET /admin/votes/{vote_id}
Vote details.

#### POST /admin/votes
Create a vote (admin override).

#### POST /admin/votes/{vote_id}/cast
Cast a vote.

### Ledger

#### GET /admin/ledger/root
Ledger root hash for verification.

**Response:**
```json
{
  "root_hash": "abc123...",
  "total_entries": 1000,
  "last_updated": 1234567890
}
```

#### GET /admin/ledger/verify/{index}
Verify a ledger entry at index.

**Response:**
```json
{
  "index": 100,
  "valid": true,
  "hash": "xyz789...",
  "previous_hash": "abc123..."
}
```

### Admin Statistics

#### GET /admin/multipliers/stats
Credit multiplier statistics.

#### GET /admin/rebenchmark/stats
Re-benchmarking statistics.

#### GET /admin/sybil/stats
Sybil detection statistics.

#### GET /admin/sybil/nodes/{node_id}
Sybil report for a specific node.

#### GET /admin/transfers/stats
Credit transfer statistics.

## Public API (v1)

### POST /v1/nodes/register
Register a new node.

**Request Body:**
```json
{
  "node_id": "optional-custom-id",
  "public_key": "",
  "invite_code": "optional",
  "os": "Linux",
  "gpu_model": "RTX 4090",
  "driver_version": "535.104"
}
```

**Response:**
```json
{
  "node_id": "node-abc123",
  "jwt": "eyJ...",
  "jwt_expires_in": 21600,
  "recommended_region": "us-east-1",
  "benchmark_required": true
}
```

### POST /v1/nodes/register-enhanced
Enhanced registration with PoC and Sybil detection.

**Request Body:**
```json
{
  "challenge_id": "challenge-123",
  "node_id": "optional",
  "os": "Linux",
  "gpu_model": "RTX 4090",
  "driver_version": "535.104",
  "invite_code": "optional"
}
```

### POST /v1/nodes/challenge
Request a PoC challenge.

**Response:**
```json
{
  "challenge_id": "challenge-123",
  "difficulty": 4,
  "prefix": "abc",
  "deadline": 1234567950
}
```

### POST /v1/nodes/challenge/verify
Verify PoC solution.

### POST /v1/jobs
Create a job (authenticated).

### GET /v1/jobs/{job_id}
Job status.

**Response:**
```json
{
  "job_id": "job-abc123",
  "status": "running",
  "progress_pct": 45,
  "current_step": 45,
  "total_steps": 100,
  "active_nodes": 3,
  "eta_seconds": 1200
}
```

### GET /v1/queue
Public job queue.

**Response:**
```json
{
  "jobs": [...],
  "active_p0_jobs": 2,
  "network_utilization_pct": 75.5
}
```

### POST /v1/votes
Cast a vote for a job.

**Request Body:**
```json
{
  "job_id": "job-abc123",
  "credits": 100
}
```

**Response:**
```json
{
  "vote_id": "vote-xyz789",
  "credits_deducted": 100,
  "job_new_vote_total": 250,
  "job_new_queue_position": 3,
  "your_new_balance": 1400,
  "trust_score": 0.95
}
```

### GET /v1/votes
List votes.

### GET /v1/credits/balance
Credit balance.

**Response:**
```json
{
  "confirmed": 1400,
  "pending": 0,
  "lifetime_earned": 2000,
  "lifetime_votes_cast": 5
}
```

### POST /v1/credits/transfer
Transfer credits to another node.

**Request Body:**
```json
{
  "to_node_id": "node-xyz789",
  "amount": 100,
  "reason": "Team pooling"
}
```

### GET /v1/credits/transfers
Transfer history.

### GET /v1/credits/multipliers
Credit multiplier status.

**Response:**
```json
{
  "base_multiplier": 1.0,
  "reliability_multiplier": 1.25,
  "early_adopter_multiplier": 1.0,
  "surge_multiplier": 1.0,
  "effective_multiplier": 1.25,
  "total_earned": 2000
}
```

### POST /v1/credits/surge-opt-in
Enable/disable surge multiplier.

### POST /v1/nodes/benchmark-status
Check whether a re-benchmark is needed. Supply the node JWT as a Bearer token.

### POST /v1/nodes/benchmark
Submit benchmark results. `overall_score` must be finite and in the inclusive range `0..100`; supply the node JWT as a Bearer token.

### POST /api/admin/rebenchmark/trigger
Ask every connected node (or one node via `{ "node_id": "..." }`) to run a real benchmark. The orchestrator sends `ControlMessage(action="benchmark")`; a busy worker defers until its current task finishes. The response includes `scheduled` and `node_ids`.

## Rate Limiting

All endpoints use a token-bucket rate limiter:
- **Default**: 10 requests per second per IP
- **Authentication endpoints**: 5 requests per minute
- **Job creation**: 20 requests per minute

Rate limit headers:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 8
X-RateLimit-Reset: 1234567895
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | Bad request format |
| `unauthorized` | 401 | Missing or invalid JWT |
| `forbidden` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `rate_limited` | 429 | Too many requests |
| `internal_error` | 500 | Server error |

## WebSocket Events (SSE)

The orchestrator exposes Server-Sent Events at **`GET /admin/stream`** for live admin updates (see `services_python/sse_limits.py` for connection limits). Express dashboards proxy this stream from the orchestrator admin URL configured at startup.

```
event: node_connected
data: {"node_id": "node-123", "timestamp": 1234567890}

event: job_created
data: {"job_id": "job-abc", "status": "queued"}

event: job_assigned
data: {"job_id": "job-abc", "node_id": "node-123"}

event: job_completed
data: {"job_id": "job-abc", "status": "success"}

event: credits_earned
data: {"node_id": "node-123", "amount": 100}
```
