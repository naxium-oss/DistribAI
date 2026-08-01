# Server Operator Guide

Guide for running DistribAI Server and managing a compute grid.

## Prerequisites

- Server with public IP or domain
- Python 3.11+ (for source install)
- Or use pre-built package

Recommended specs:
- 4+ CPU cores
- 8+ GB RAM
- 50+ GB SSD
- 100+ Mbps network

## Installation

### Option 1: Pre-built Package

Download for your platform:
```bash
# Windows: DistribAI-Server-Windows-Setup.exe
# macOS: DistribAI-Server-macos.app
# Linux: DistribAI-Server-Linux.tar.gz
```

### Option 2: Source

```bash
git clone https://github.com/naxium-oss/DistribAI.git
cd distribai
pip install -r requirements.txt
```

## Initial Setup

### Preview the dashboard before running a grid

To inspect the GUI without packaging or starting server/worker processes:

```bash
npm install
npm run gui:preview
```

This opens `http://127.0.0.1:3210/?role=admin` with deterministic preview data. Use `python scripts/dev/preview_gui.py --role node` to inspect the node/contributor view.

Browser coverage uses the same preview mode:

```bash
npx playwright install chromium
npm run test:ui
```

### 1. Launch Server

Pre-built:
```bash
./DistribAI-Server  # macOS/Linux
DistribAI-Server.exe  # Windows
```

Source:
```bash
python -m services_python.server_gui
```

### 2. Configure Settings

First launch opens Settings panel:

**Network:**
- gRPC Port: 50051 (for nodes)
- Admin Port: 8766 (for dashboard)
- Admin Host: 0.0.0.0 (for remote access) or 127.0.0.1 (local only)

**Security:**
- JWT Secret: Auto-generated
- Signing Key: For ledger integrity

**Database:**
- SQLite: Default, no config needed
- PostgreSQL: Enter connection string for larger grids

**Storage (optional):**
- S3 credentials for checkpoint persistence

**Update Hosting:**
- URL where node packages are hosted
- Nodes check this for updates

Click **Save Settings**.

### 3. Start Server

Click **Start Server** button.

Status should show: 🟢 Running

### 4. Verify Operation

```bash
curl http://localhost:8766/admin/health
```

Expected response:
```json
{"ok": true, "active_nodes": 0, "queued_jobs": 0}
```

## Managing the Grid

### Dashboard Overview

**Status Bar:**
- Connected Nodes: Currently online
- Active Jobs: Jobs with pending tasks
- Total Jobs: All-time job count
- Credits Issued: Total distributed

**Tabs:**
- Dashboard: Overview and controls
- Nodes: Connected contributors
- Jobs: Training jobs
- Ledger: Credit distribution
- Settings: Configuration

### Creating Jobs

#### Via GUI

1. Click **Jobs** tab
2. Click **Create Job**
3. Fill details:
   - Name: Descriptive identifier
   - Model: Base model or custom
   - Dataset: Training data reference
   - Steps: Total training iterations
   - Priority: P0 (critical) to P3 (low)
4. Click **Create**

Job appears in queue and nodes auto-pickup.

#### Via API

```bash
curl -X POST http://localhost:8766/admin/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT" \
  -d '{
    "job_type": "fine_tune",
    "base_model": "DistribAI/DistribAI-SLM-300M",
    "dataset_ref": "wikitext-103",
    "steps": 10000,
    "priority": "P1"
  }'
```

### Managing Nodes

**Nodes tab shows:**
- Node ID: Anonymous identifier
- Status: Online/Offline/Working
- GPU: Hardware type
- Credits: Lifetime earnings
- Reliability: Score 0-100

**Actions:**
- Click node for details
- Ban: Exclude malicious nodes
- Reward Boost: Increase multiplier
- View History: Completed tasks

### Byzantine Detection

Automatic protection against malicious nodes:

**Methods Used:**
- Coordinate-wise Median
- Trimmed Mean
- Multi-Krum
- Clustering
- AUROR
- Reputation tracking

**Settings:**
Settings → Security → Byzantine Tolerance
- Threshold: % of nodes that can be malicious
- Default: 33% (handles up to 1/3 bad actors)

## Credit Ledger

### How Credits Work

1. Node completes task → Submits gradient
2. Gradient passes validation
3. Credit issued to node
4. Record added to ledger
5. Merkle root updated

### Ledger Features

- **Tamper-proof**: Cryptographic chain
- **Verifiable**: Anyone can check integrity
- **Transparent**: All transactions public

**Ledger tab shows:**
- Total records
- Merkle root hash
- Recent transactions

**Verification:**
Click **Verify Integrity** to check chain.

## Scaling

### SQLite (Default)

Handles:
- 1,000 nodes
- 100,000 jobs
- Single server

### PostgreSQL (Large Grids)

Settings → Database → PostgreSQL

Connection string format:
```
postgresql://user:password@host:5432/database
```

Handles:
- 10,000+ nodes
- 1,000,000+ jobs
- Multi-server deployment

### Multiple Servers

For geographic distribution:

1. Deploy servers in regions
2. Each with own database
3. Share job queue via message bus
4. Use load balancer for nodes

## Security

### Network Security

**Firewall Rules:**
```bash
# Allow node connections
iptables -A INPUT -p tcp --dport 50051 -j ACCEPT

# Allow admin dashboard (restrict to your IP)
iptables -A INPUT -p tcp -s YOUR_IP --dport 8766 -j ACCEPT
```

**TLS (optional):**
Settings → Security → Enable TLS
- Upload certificate and key
- Nodes connect via TLS

### Authentication

**JWT Tokens:**
- Auto-generated on first run
- Rotate periodically
- Stored in `.env`

**API Keys:**
Generate for external integrations:
Settings → Security → API Keys → Generate

### Access Control

**Admin Dashboard:**
- Bind to 127.0.0.1 for local-only
- Or use firewall rules
- Enable auth if exposed

## Monitoring

### Built-in Metrics

Dashboard shows real-time:
- Node count
- Job throughput
- Credit issuance rate
- Byzantine detection triggers

### Admin API Metrics

Poll the admin API or use the desktop dashboard:
```bash
curl http://localhost:8766/admin/health
curl http://localhost:8766/admin/nodes
curl http://localhost:8766/admin/jobs
```

### Alerts

Setup notifications for:
- Node count drops
- Job queue backs up
- Byzantine detection fires
- Ledger issues

## Backup and Recovery

### Database Backup

**SQLite:**
```bash
cp distribai.db distribai_backup_$(date +%Y%m%d).db
```

**PostgreSQL:**
```bash
pg_dump -U user distribai > backup.sql
```

### Ledger Backup

Critical - must not lose:
```bash
cp -r ledger_data/ ledger_backup/
```

### Recovery

1. Stop server
2. Restore database
3. Restore ledger (critical)
4. Start server
5. Verify integrity

## Troubleshooting

### "Port already in use"

Find and kill process:
```bash
# Linux/macOS
lsof -i :50051
kill -9 <PID>

# Windows
netstat -ano | findstr :50051
taskkill /PID <PID> /F
```

Or change ports in Settings.

### "Database locked"

SQLite doesn't support multiple writers:
- Ensure only one server instance
- Check for zombie processes

### "Cannot bind to 0.0.0.0"

Firewall or permission issue:
```bash
# Check permissions
python -m services_python.server_gui

# Or bind to specific IP
# Settings → Admin Host → Your IP
```

### Nodes not connecting

1. Check server is running
2. Verify port 50051 is open
3. Check firewall rules
4. Test from node: `telnet server_ip 50051`

### Job not progressing

1. Check active nodes count
2. Verify job priority
3. Check Byzantine flags (nodes may be excluded)
4. Review job logs

## Update Hosting

### Setup

Nodes need to download updates from you.

**Option 1: GitHub Releases (Free)**

1. Create public repo: `yourname/distribai-releases`
2. Upload built packages to Releases
3. Set in Settings:
   - Update Hosting URL: `https://github.com/yourname/distribai-releases`

**Option 2: Self-hosted**

1. Upload packages to web server
2. Create `version.json`:
```json
{
  "version": "1.1.0",
  "download_url": "https://your-server.com/DistribAI-Node-Windows.exe",
  "size_mb": 450,
  "hash": "sha256:abc123...",
  "notes": "Bug fixes"
}
```
3. Set URL in Settings

### Release Process

1. Build new packages: `python setup.py --build-only` (or `python build.py all` for wheels only)
2. Test locally
3. Upload to hosting
4. Update `version.json`
5. Nodes auto-detect on next check

## Advanced Configuration

### Environment Variables

All settings can be set via `.env`:

```bash
# Network
GRPC_PORT=50051
ADMIN_PORT=8766
ADMIN_HOST=0.0.0.0

# Security
JWT_SECRET=your-secret
SIGNING_KEY=your-signing-key

# Database
POSTGRES_URL=postgresql://...

# Storage
S3_BUCKET_NAME=your-bucket
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx

# Updates
UPDATE_HOSTING_URL=https://your-server.com

# Tuning
MAX_NODES=1000
BATCH_TIMEOUT_SECONDS=30
```

### Command Line

```bash
# Custom ports
GRPC_PORT=50052 ADMIN_PORT=8767 python -m services_python.server_gui

# Headless (no GUI)
python -m services_python.orchestrator_grpc
```

## Support

- **Issues (primary support)**: https://github.com/naxium-oss/DistribAI/issues
- **Documentation**: https://docs.distribai.io
