# Incident Response Runbook

**Last Updated:** 2026-04-29  
**Priority:** P0 (Critical) - Must be followed during any incident

---

## Quick Reference Cards

### Severity Levels

| Level | Name | Response Time | Example |
|-------|------|---------------|---------|
| SEV1 | Critical | 15 min | All nodes offline, data loss |
| SEV2 | High | 1 hour | >50% nodes down, credit system failure |
| SEV3 | Medium | 4 hours | Gradual node attrition, partial S3 outage |
| SEV4 | Low | 24 hours | Dashboard glitch, minor UI issues |

### Emergency Contacts

```
Primary On-Call:    See PagerDuty rotation
Escalation:        DistribAI Infrastructure Team
Security Issues:   https://github.com/naxium-oss/DistribAI/security/advisories/new
Discord Alerts:    #incident-response
```

---

## Incident Response Lifecycle

### 1. Detection

**Automated Alerts (PagerDuty):**
- Node count drops >20% in 5 minutes
- Database connection failures
- S3 upload failures >5%
- Credit ledger inconsistencies detected
- Byzantine node rate >10%

**Manual Detection:**
- Community reports on Discord
- Dashboard anomalies
- Training job failures

### 2. Response

**First 5 Minutes:**
1. Acknowledge alert in PagerDuty
2. Join #incident-response Discord channel
3. Post initial message: `Investigating [SEV-X] incident: [brief description]`
4. Check status page: `https://github.com/naxium-oss/DistribAI/issues`

**First 15 Minutes (SEV1):**
1. Assess scope: How many nodes affected? Which regions?
2. Identify if security-related (freeze all credit transfers)
3. Check recent deployments: `git log --oneline -10`
4. Review orchestrator logs: `tail -f /var/log/distribai/orchestrator.log`

### 3. Containment

**Immediate Actions by Incident Type:**

**Node Mass Disconnection:**
```bash
# Check if orchestrator is healthy
curl http://localhost:8766/admin/health

# Review recent node disconnections
sqlite3 /path/to/distribai.db \
  "SELECT node_id, status, last_heartbeat_ts FROM active_nodes WHERE status='offline' ORDER BY last_heartbeat_ts DESC LIMIT 50;"

# If DDoS suspected, enable rate limiter emergency mode
export RATE_LIMIT_EMERGENCY_MODE=1
systemctl restart distribai-orchestrator
```

**Credit System Anomaly:**
```bash
# Immediately freeze credit transfers
curl -X POST http://localhost:8766/admin/freeze-credits \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Run ledger consistency check
python -m worker.src.daemon.credit_ledger --verify-chain

# Generate forensic report
python scripts/generate_ledger_report.py --since $(date -d '1 hour ago' +%s)
```

**Byzantine Attack Detected:**
```bash
# Enable strict aggregation mode
export BYZANTINE_MODE=strict
systemctl restart distribai-orchestrator

# Isolate suspicious nodes
sqlite3 /path/to/distribai.db \
  "UPDATE active_nodes SET status='quarantined' WHERE reliability_score < 0.3;"
```

### 4. Investigation

**Log Collection:**
```bash
# Collect all relevant logs
mkdir -p /tmp/incident-$(date +%Y%m%d-%H%M%S)
journalctl -u distribai-orchestrator --since "1 hour ago" > /tmp/incident-*/orchestrator.log
journalctl -u distribai-worker --since "1 hour ago" > /tmp/incident-*/worker.log
sqlite3 /path/to/distribai.db ".dump" > /tmp/incident-*/database.sql
```

**Key Metrics to Check:**
- Node count over time: `SELECT COUNT(*) FROM active_nodes WHERE status='online'`
- Job success rate: Last hour vs baseline
- Credit transaction volume: Sudden spikes?
- Gradient anomaly rate: `SELECT COUNT(*) FROM tasks WHERE status='invalid_gradient'`

### 5. Recovery

**Gradual Rollback Strategy:**
1. Deploy fix to staging environment
2. Test with 5% of nodes
3. Monitor for 30 minutes
4. Expand to 25% of nodes
5. Full rollout after 2 hours of stability

**Verification Steps:**
```bash
# Verify orchestrator health
curl -f http://localhost:8766/admin/health || echo "HEALTH CHECK FAILED"

# Verify database connectivity
python -c "from services_python.db_manager import DBManager; db = DBManager('...', '...'); print(db.get_stats())"

# Verify S3 connectivity
aws s3 ls s3://$S3_BUCKET_NAME/checkpoints/ --max-items 5

# Spot check node registrations
for node in $(curl -s http://localhost:8766/admin/nodes | jq -r '.nodes[0:5].node_id'); do
  echo "Checking $node..."
  curl -s http://localhost:8766/admin/nodes/$node | jq '.status'
done
```

### 6. Post-Incident

**Required Within 24 Hours:**
- Post-incident review meeting scheduled
- Initial timeline documented
- Preliminary root cause identified

**Required Within 1 Week:**
- Full postmortem published
- Action items assigned with owners
- Monitoring improvements deployed
- Runbook updates merged

---

## Common Incident Types

### Type A: Orchestrator Crash During Training

**Symptoms:**
- All nodes report "connection refused"
- Active jobs show "timeout" status
- Training progress stalled

**Immediate Actions:**
```bash
# 1. Check orchestrator process
ps aux | grep orchestrator_grpc

# 2. If not running, start with recovery flag
export RECOVERY_MODE=1
python -m services_python.orchestrator_grpc &

# 3. Re-queue incomplete tasks
sqlite3 /path/to/distribai.db \
  "UPDATE tasks SET status='queued' WHERE status='assigned' AND updated_ts < $(($(date +%s) - 600));"
```

**Root Cause Analysis:**
- Memory exhaustion? Check `dmesg | grep -i "out of memory"`
- Database corruption? Run `PRAGMA integrity_check;`
- Gradient aggregation overload? Check recent job batch sizes

### Type B: Byzantine Node Flood

**Symptoms:**
- Invalid gradient rate >15%
- Reputation scores dropping across many nodes
- Training convergence stalled

**Immediate Actions:**
```bash
# 1. Switch to most conservative aggregation
export AGGREGATION_METHOD=coordinate_median
export BYZANTINE_THRESHOLD=0.1
systemctl restart distribai-orchestrator

# 2. Identify and quarantine suspicious nodes
python scripts/quarantine_byzantine_nodes.py --threshold 0.5

# 3. Enable additional verification layers
export VERIFY_GRADIENT_HASHES=1
```

### Type C: Credit System Exploit

**Symptoms:**
- Unusual credit balance patterns
- Multiple accounts with identical fingerprints
- Vote manipulation detected

**Immediate Actions:**
```bash
# 1. EMERGENCY: Freeze all credit operations
curl -X POST http://localhost:8766/admin/emergency-freeze \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"reason": "Investigating potential exploit", "duration_hours": 24}'

# 2. Generate full ledger audit
python -m worker.src.daemon.credit_ledger --audit --full-chain > /tmp/ledger_audit.json

# 3. Check for sybil patterns
python scripts/detect_sybil_clusters.py --time-window-hours 24

# 4. If exploit confirmed, prepare rollback
cp /path/to/distribai.db /path/to/distribai-emergency-backup-$(date +%s).db
```

### Type D: S3 Storage Outage

**Symptoms:**
- Gradients failing to upload
- Checkpoint retrieval timeouts
- Model weights inaccessible

**Immediate Actions:**
```bash
# 1. Switch to local storage fallback
export STORAGE_MODE=local
export LOCAL_STORAGE_PATH=/mnt/emergency-storage
mkdir -p $LOCAL_STORAGE_PATH

# 2. Notify users of degraded mode
curl -X POST http://localhost:8766/admin/broadcast \
  -d '{"message": "Storage degraded. Checkpoints temporarily local.", "severity": "warning"}'

# 3. Monitor disk space
df -h $LOCAL_STORAGE_PATH
```

---

## Communication Templates

### Initial Incident Post (Discord/Status Page)

```
🚨 [SEV-X] Incident: [Brief Title]

Status: Investigating
Impact: [e.g., "50% of nodes experiencing disconnections"]
Started: [Timestamp UTC]

We are investigating an issue affecting [component]. 
Updates every 15 minutes or upon significant change.

Incident ID: INC-YYYYMMDD-XXX
```

### Resolution Post

```
✅ Resolved: [Incident Title]

Duration: [X minutes/hours]
Resolution: [Brief description of fix]

All systems operational. Postmortem to follow within 24 hours.
```

---

## Tooling

### Emergency Scripts

**Force Node Reconnect:**
```bash
#!/bin/bash
# force_reconnect.sh - Trigger all nodes to reconnect

curl -X POST http://localhost:8766/admin/broadcast \
  -d '{"action": "reconnect", "reason": "orchestrator_restart"}'
```

**Emergency Ledger Verification:**
```bash
#!/bin/bash
# verify_ledger.sh - Quick ledger health check

python << 'EOF'
from worker.src.daemon.credit_ledger import CreditLedger
import sys

ledger = CreditLedger()
if ledger.verify_chain(0, ledger.head.size):
    print("✅ Ledger chain valid")
    sys.exit(0)
else:
    print("❌ Ledger corruption detected!")
    sys.exit(1)
EOF
```

**Database Integrity Check:**
```bash
#!/bin/bash
# check_db.sh

sqlite3 /path/to/distribai.db << 'EOF'
SELECT 'Active nodes: ' || COUNT(*) FROM active_nodes WHERE status='online';
SELECT 'Queued jobs: ' || COUNT(*) FROM jobs WHERE status='queued';
SELECT 'Running jobs: ' || COUNT(*) FROM jobs WHERE status='running';
SELECT 'Tasks last hour: ' || COUNT(*) FROM tasks WHERE created_ts > strftime('%s', 'now') - 3600;
EOF
```

---

## Prevention Checklist

Deploy these to prevent common incidents:

- [ ] Database backups every hour (automated)
- [ ] Credit ledger checkpoints every 100 transactions
- [ ] Circuit breaker on S3 operations (fail fast after 3 errors)
- [ ] Byzantine detection auto-escalation
- [ ] Health checks on all services
- [ ] Log rotation with 30-day retention
- [ ] Automated security scanning (daily)

---

**Document Owner:** DistribAI Infrastructure Team  
**Review Schedule:** Monthly, or after every SEV1 incident

*Remember: Safety first. When in doubt, freeze and investigate.*
