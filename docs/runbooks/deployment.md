# Deployment runbook

<p style="max-width:42rem;margin:0 auto 1rem;line-height:1.55;text-align:center;">
  <a href="../../README.md"><strong>README</strong></a> &nbsp;|&nbsp; <span>(overview)</span><br/>
  <a href="../../TODO.md"><strong>TODO</strong></a> &nbsp;|&nbsp; <span>(backlog)</span><br/>
  <a href="../../AGENTS.md"><strong>AGENTS</strong></a> &nbsp;|&nbsp; <span>(rules)</span><br/>
  <a href="../README.md"><strong>Documentation index</strong></a> &nbsp;|&nbsp; <span>(all docs)</span><br/>
  <a href="deployment.md"><strong>Deploy runbook</strong></a> &nbsp;|&nbsp; <span>(this page)</span>
</p>

---
## Production release gate

Before tagging or promoting a release, run the steps in [production-release-checklist.md](production-release-checklist.md) (`npm run verify:production`, newcomer test, `verify_setup.py`).

## Local smoke (three processes)

Use this on a developer machine before promotion. systemd, TLS, backups, and rollback appear later in this runbook.

**Prerequisites:** Python 3.11+, Node.js 18+ (honor `package.json` engines when present), repository checked out.

```bash
pip install -r requirements.txt
npm install                             # dashboards and npm helpers only
cp .env.example .env                    # edit for production secrets
```

**Processes (three terminals):**

1. Orchestrator — wait until gRPC and admin bind logs appear.

   ```bash
   python -m services_python.orchestrator_grpc
   ```

2. Contributor dashboard (Express) — default `PORT=3000`.

   ```bash
   node client/server.js
   ```

3. Operator dashboard (Express) — default `ORCH_PORT=3212`.

   ```bash
   node client/orchestrator-server.js
   ```

**Quick HTTP checks:**

```bash
curl -sf http://127.0.0.1:8766/admin/health
curl -sf http://127.0.0.1:3000/api/worker/status
curl -sf http://127.0.0.1:3212/api/status
```

**Diagnostics:**

```bash
python scripts/cli/distribai_cli.py health
```

**Dashboard URLs:**

- Contributor UI — `http://127.0.0.1:3000/` (pages under `worker/src/dashboard/static/node/`, shared assets at `/shared/`).
- Operator UI — `http://127.0.0.1:3212/` (pages under `.../static/orch/`; `/node/` remounts contributor pages when operators need those links).

Mint strong secrets into `.env` (for example `python -c "import secrets; print(secrets.token_urlsafe(64))"`). Cross-check [`.env.example`](../../.env.example).

---

## Secured remote orchestrator (required for internet-facing hosts)

Do not expose the admin API on a public interface unless **every** item below is true:

| Requirement | Environment / config |
|-------------|----------------------|
| Stable signing + session secrets | `SIGNING_KEY`, `JWT_SECRET` set in `.env` (not auto-generated per boot) |
| Admin Bearer lockdown | `ADMIN_HOST` not loopback **or** `ADMIN_REQUIRE_AUTH=1`, plus dedicated `DISTRIBAI_ADMIN_SECRET` (do not reuse `JWT_SECRET`) |
| Dashboard proxies | Same `DISTRIBAI_ADMIN_SECRET` for `client/server.js` and `client/orchestrator-server.js` |
| Transport | `GRPC_USE_TLS=true` with valid certs **or** traffic confined to VPN/private network |
| Firewall | Only operator IPs reach admin HTTP and gRPC ports |

**Verification after deploy:**

```bash
# Must succeed without auth (health probe)
curl -sf http://<host>:8766/admin/health

# Must return 401 without Bearer when lockdown is active
curl -s -o /dev/null -w "%{http_code}" http://<host>:8766/admin/jobs
# expect 401

# Must succeed with secret
curl -sf -H "Authorization: Bearer $DISTRIBAI_ADMIN_SECRET" http://<host>:8766/admin/jobs
```

Loopback development (`127.0.0.1`) leaves admin routes open unless `ADMIN_REQUIRE_AUTH=1` is set.

**Mini smoke (orchestrator already running):**

```bash
npm run smoke:mini
# or: python -m scripts.dev.mini_smoke --admin-url http://127.0.0.1:8766
```

Public bind without stable secrets or TLS is **refused at startup** unless `ALLOW_INSECURE_PUBLIC_BIND=1` (private lab only).

### TLS / mTLS certificate rotation

| Step | Action |
|------|--------|
| 1 | Issue new server cert + key (`GRPC_TLS_CERT`, `GRPC_TLS_KEY`) and worker client cert + key when using mTLS (`GRPC_TLS_CLIENT_CERT`, `GRPC_TLS_CLIENT_KEY`). |
| 2 | Distribute updated `GRPC_TLS_CA` trust bundle to all workers before cutover. |
| 3 | Rolling restart orchestrator with new server material; verify `curl -sf https://<host>:8766/admin/health` via nginx or direct TLS admin proxy. |
| 4 | Restart workers with new client certs; confirm gRPC stream reconnects and `/api/operator/status` shows `ledger_sql_memory_drift_count: 0` after steady-state traffic. |
| 5 | Revoke or destroy superseded private keys; never commit `.env` or PEM files to git. |

Set `GRPC_TLS_REQUIRE_CLIENT_CERT=true` only when every worker presents a client certificate signed by the CA in `GRPC_TLS_CA`.

### Artifact egress by deployment profile

| Profile | Gradients / weights | Script bundles | Credit ledger |
|---------|---------------------|----------------|---------------|
| Local dev (`127.0.0.1`) | Local paths under `runtime/` or allowed `ALLOWED_BLOB_HOSTS` | `runtime/bundles/{task_id}.tar.gz` (optional S3 mirror) | SQLite file under `runtime/db/` |
| Single-host prod | S3 bucket from `S3_BUCKET_NAME` + allowlisted hosts | Same; workers fetch via scheduler assign | SQLite on orchestrator disk; back up before upgrades |
| Multi-instance (future) | Shared S3 required; SQLite not shared across orchestrators | S3 mirror required for script assign | One SQLite writer per orchestrator process |

Authoritative route list: [`docs/api/endpoints.md`](../api/endpoints.md) (Artifact egress table).

---

## Prerequisites (reference)

### System Requirements

**Orchestrator Host:**
- Python 3.11+
- 8GB RAM minimum (16GB recommended)
- 50GB disk space
- Public IP or domain
- SSL certificates (for production TLS)

**Worker Nodes:**
- Python 3.11+
- CUDA-capable GPU (recommended)
- 4GB RAM minimum
- 20GB disk space
- Stable internet connection

### Environment Setup

```bash
# Clone repository
git clone https://github.com/your-org/distribai.git
cd distribai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (repo root only — worker shares requirements.txt)
pip install -r requirements.txt
```

## Orchestrator Deployment

### 1. Configuration

Create a `.env` file:

```bash
# Core Settings
GRPC_PORT=50051
ADMIN_PORT=8766
ADMIN_HOST=0.0.0.0

# Security (auto-generated if not set)
JWT_SECRET=your-secure-random-secret-here
SIGNING_KEY=your-ledger-signing-key-here
DISTRIBAI_ADMIN_SECRET=your-admin-bearer-secret-here

# SQLite (orchestrator picks runtime/db/distribai-<grpc>-<admin>.db automatically)
# Do not set DB_PATH — not read by services_python/database.py

# S3 Configuration
S3_BUCKET_NAME=your-bucket
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1

# Optional: TLS
GRPC_USE_TLS=true
GRPC_TLS_CERT=./certs/server.crt
GRPC_TLS_KEY=./certs/server.key

# Optional: Redis (falls back to in-memory)
REDIS_URL=redis://localhost:6379/0

# Optional: CORS
CORS_ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

### 2. Database Initialization

```bash
# Database is auto-initialized on first run
# Manual initialization (optional):
python -c "
from services_python.db_manager import DBManager
db = DBManager('./runtime/db/distribai.db', './runtime/db/schema.sql')
print('Database initialized')
"
```

### 3. Start Orchestrator

```bash
# Development
python -m services_python.orchestrator_grpc

# Production (with logging)
python -m services_python.orchestrator_grpc 2>&1 | tee /var/log/distribai.log

# Using systemd (see systemd service file below)
systemctl start distribai-orchestrator
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8766/admin/health

# Expected response:
# {"ok": true, "timestamp": 1234567890, "version": "1.0.0"}

# Create test job
curl -X POST http://localhost:8766/admin/jobs \
  -H "Content-Type: application/json" \
  -d '{"steps": 10, "batch_size": 8}'

# List jobs
curl http://localhost:8766/admin/jobs
```

## Worker Deployment

### 1. Configuration

Set environment variables:

```bash
export ORCHESTRATOR_URL=your-orchestrator-host:50051
export ADMIN_URL=http://your-orchestrator-host:8766
export DISTRIBAI_JWT_TOKEN=optional-pre-generated-token
export DISTRIBAI_INVITE_CODE=optional-invite-code

# Optional: API URL override (if admin is not on default port/path)
export DISTRIBAI_API_URL=http://your-orchestrator-host:8766

# Optional: TLS to orchestrator (required on public grids)
export GRPC_USE_TLS=true
export GRPC_TLS_CA=/path/to/orchestrator-ca.pem

# Optional: Colab/Kaggle burst workers
export DISTRIBAI_EPHEMERAL=1

# Optional: Benchmark blocking
export DISTRIBAI_BLOCK_ON_BENCHMARK=false

# Optional: State directory (CLI: --state-dir)
export STATE_DIR=./runtime
```

### 2. Run Worker

```bash
# Run single worker
python -m worker.src.daemon.run

# Run multiple workers (on same machine for testing)
python -m tools.launch_workers --count 3
```

### 3. Verify Worker

```bash
# Check orchestrator nodes list
curl http://localhost:8766/admin/nodes

# Worker should appear in list
```

## Desktop Application Deployment

DistribAI ships desktop apps for both roles:

- Server operators run the server desktop app (`python -m services_python.server_gui` from
  source, or the packaged Windows installer / macOS `.app`).
- Node contributors run the node worker (`python -m worker.src.daemon.run` from
  source, or the packaged Windows installer / macOS `.app`).

Experimental Kubernetes / Helm packaging lives under
[`deploy/helm/distribai/`](../../deploy/helm/distribai/). Read the caveats in
[kubernetes.md](kubernetes.md) before using it: SQLite remains single-owner
(keep orchestrator replicas at 1), Redis is recommended for multi-replica
coordination, and you must supply your own container image. systemd unit
examples below remain the primary documented VM path.

## systemd Service Files

### Orchestrator Service

```ini
# /etc/systemd/system/distribai-orchestrator.service
[Unit]
Description=DistribAI Orchestrator
After=network.target

[Service]
Type=simple
User=distribai
Group=distribai
WorkingDirectory=/opt/distribai
Environment=PYTHONPATH=/opt/distribai
EnvironmentFile=/opt/distribai/.env
ExecStart=/opt/distribai/venv/bin/python -m services_python.orchestrator_grpc
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable distribai-orchestrator
sudo systemctl start distribai-orchestrator
sudo systemctl status distribai-orchestrator
```

### Worker Service

```ini
# /etc/systemd/system/distribai-worker.service
[Unit]
Description=DistribAI Worker
After=network.target

[Service]
Type=simple
User=distribai-worker
Group=distribai-worker
WorkingDirectory=/opt/distribai
Environment=PYTHONPATH=/opt/distribai
Environment=DISTRIBAI_ORCHESTRATOR_HOST=orchestrator.example.com
ExecStart=/opt/distribai/venv/bin/python -m worker.src.daemon.run
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## SSL/TLS Setup

### Generate Self-Signed Certificates (Development)

```bash
mkdir -p certs
cd certs

# Generate private key
openssl genrsa -out server.key 2048

# Generate certificate
openssl req -new -x509 -key server.key -out server.crt -days 365 \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Set permissions
chmod 600 server.key
chmod 644 server.crt
```

### Let's Encrypt (Production)

```bash
# Install certbot
sudo apt-get install certbot

# Obtain certificate
sudo certbot certonly --standalone -d your-domain.com

# Update .env
GRPC_TLS_CERT=/etc/letsencrypt/live/your-domain.com/fullchain.pem
GRPC_TLS_KEY=/etc/letsencrypt/live/your-domain.com/privkey.pem
GRPC_TLS_CA=/etc/letsencrypt/live/your-domain.com/fullchain.pem
```

Workers must set the same `SIGNING_KEY` as the orchestrator when verifying ledger-linked credits locally.

### HTTPS reverse proxy for the admin API (production)

Terminate TLS at nginx (or Caddy) and proxy to loopback admin HTTP. Keep `ADMIN_HOST=127.0.0.1` on the orchestrator process so the admin socket is not exposed without the proxy.

```nginx
server {
    listen 443 ssl http2;
    server_name orch.example.com;

    ssl_certificate     /etc/letsencrypt/live/orch.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/orch.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8766;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Dashboard clients (`client/server.js`, `client/orchestrator-server.js`) discover the orchestrator admin API via `ADMIN_HOST` + `ADMIN_PORT` (see `.env.example`). Put HTTPS in front with a reverse proxy (example above); set the same `DISTRIBAI_ADMIN_SECRET` on both Express servers so proxied admin calls include `Authorization: Bearer …`. gRPC remains on `GRPC_PORT` with `GRPC_USE_TLS=true` and matching certs; workers set `GRPC_TLS_CA` to the orchestrator CA bundle.

## SQLite single-owner expectations

The orchestrator uses **one writable SQLite file per process** (under `runtime/db/`, often named `distribai-<grpc>-<admin>.db`). This is intentional for single-node and dev deployments—not a multi-writer HA cluster.

| Expectation | Guidance |
|-------------|----------|
| **Concurrency** | One orchestrator process owns the DB file. Do not NFS-share the same `.db` across hosts. |
| **Failover** | Stop orchestrator → restore `.backup` file → start. There is no automatic leader election. |
| **Replication** | Use scheduled `.backup` copies (below) or volume snapshots; do not commit `*.db` to git. |
| **Schema** | Source of truth: `runtime/db/schema.sql`; runtime may add columns via `DBManager` migrations. |
| **Hygiene** | Keep zero–three active DB files locally; delete stale test DBs after integration runs. |

For production fleets needing hot standby, plan an external database or orchestrator-per-shard model before promising five-nines uptime.

## Backup Procedures

### Database Backup

```bash
# SQLite backup
cp runtime/db/distribai.db runtime/db/distribai-backup-$(date +%Y%m%d).db

# Or using SQLite command
sqlite3 runtime/db/distribai.db ".backup runtime/db/backup-$(date +%Y%m%d).db"

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backup/distribai"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"
sqlite3 /opt/distribai/runtime/db/distribai.db ".backup $BACKUP_DIR/distribai-$DATE.db"
find "$BACKUP_DIR" -name "distribai-*.db" -mtime +7 -delete
```

### S3 Backup

```bash
# Sync checkpoints
aws s3 sync s3://your-bucket/checkpoints s3://your-backup-bucket/checkpoints

# Versioning should be enabled on S3 bucket
```

## Rollback Procedures

### Database Rollback

```bash
# Stop orchestrator
sudo systemctl stop distribai-orchestrator

# Restore from backup
cp runtime/db/distribai-backup-20240101.db runtime/db/distribai.db

# Start orchestrator
sudo systemctl start distribai-orchestrator
```

### Code Rollback

```bash
# Git rollback
git log --oneline -10  # Find commit to rollback to
git checkout <commit-hash>

# Restart services
sudo systemctl restart distribai-orchestrator
```

## Monitoring Setup

### Admin API Metrics

Use the admin API and desktop dashboard for operational health:

```bash
curl http://localhost:8766/admin/health
curl http://localhost:8766/admin/nodes
curl http://localhost:8766/admin/jobs
```

### Log Aggregation

```bash
# Configure rsyslog
echo "local0.* /var/log/distribai.log" >> /etc/rsyslog.conf

# Logrotate configuration
cat > /etc/logrotate.d/distribai <<EOF
/var/log/distribai.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 distribai distribai
}
EOF
```

## Troubleshooting Deployment

### Port Conflicts

```bash
# Check if ports are in use
sudo lsof -i :50051
sudo lsof -i :8766

# Change ports in .env if needed
GRPC_PORT=50052
ADMIN_PORT=8767
```

### Permission Issues

```bash
# Fix permissions
sudo chown -R distribai:distribai /opt/distribai
sudo chmod 755 /opt/distribai/runtime
sudo chmod 644 /opt/distribai/.env
```

### Database Locks

```bash
# Check for locks
lsof runtime/db/distribai.db

# Kill hanging process if needed
kill -9 <pid>
```

## Health Check Script

```bash
#!/bin/bash
# health-check.sh

ORCHESTRATOR_URL=${1:-http://localhost:8766}

echo "Checking orchestrator health..."
if curl -sf "$ORCHESTRATOR_URL/admin/health" > /dev/null; then
    echo "✓ Orchestrator is healthy"
else
    echo "✗ Orchestrator is not responding"
    exit 1
fi

echo "Checking nodes..."
NODES=$(curl -sf "$ORCHESTRATOR_URL/admin/nodes" | jq '.nodes | length')
echo "  Connected nodes: $NODES"

echo "Checking queue..."
QUEUE=$(curl -sf "$ORCHESTRATOR_URL/admin/jobs" | jq '.queue_depth')
echo "  Queue depth: $QUEUE"

echo "All checks passed!"
```

## Post-Deployment Checklist

- [ ] Orchestrator health endpoint responds
- [ ] Worker can register successfully
- [ ] Test job completes successfully
- [ ] Credit ledger updates correctly
- [ ] S3 uploads working
- [ ] Logs are being written
- [ ] Monitoring alerts configured
- [ ] Backup jobs scheduled
- [ ] Documentation updated
- [ ] Team notified of deployment
