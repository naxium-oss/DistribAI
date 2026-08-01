# Troubleshooting Runbook

## Quick Diagnosis

### Status Check Commands

```bash
# 1. Check orchestrator health
curl http://localhost:8766/admin/health

# 2. List active nodes
curl http://localhost:8766/admin/nodes | jq '.nodes[] | {id: .node_id, status: .status}'

# 3. Check job queue
curl http://localhost:8766/admin/jobs | jq '{queue_depth: .queue_depth, running: [.jobs[] | select(.status == "running")] | length}'

# 4. View recent logs
tail -100 /var/log/distribai.log

# 5. Check system resources
free -h && df -h && nvidia-smi
```

## Common Issues

### Issue: Worker Cannot Connect

**Symptoms:**
- Worker logs show "Connection refused" or "RPC error"
- Node does not appear in `/admin/nodes`

**Diagnosis:**

```bash
# 1. Check orchestrator is running
curl http://localhost:8766/admin/health

# 2. Verify gRPC port is listening
sudo lsof -i :50051

# 3. Check firewall rules
sudo iptables -L | grep 50051

# 4. Test connectivity from worker
telnet <orchestrator-host> 50051
```

**Solutions:**

1. **Orchestrator not running:**
   ```bash
   sudo systemctl start distribai-orchestrator
   ```

2. **Firewall blocking:**
   ```bash
   sudo ufw allow 50051/tcp
   sudo ufw allow 8766/tcp
   ```

3. **Wrong host configuration:**
   ```bash
   export DISTRIBAI_ORCHESTRATOR_HOST=correct-hostname
   ```

### Issue: Jobs Stuck in Queue

**Symptoms:**
- Jobs created but never assigned
- Queue depth increases continuously
- No tasks sent to workers

**Diagnosis:**

```bash
# Check scheduler loop status
grep "scheduler\|assign" /var/log/distribai.log | tail -20

# Verify nodes are idle
curl /admin/nodes | jq '.nodes[] | select(.status == "idle")'

# Check for errors in job creation
grep -i "error\|exception" /var/log/distribai.log | tail -20
```

**Solutions:**

1. **No idle nodes:**
   - Add more workers, or
   - Wait for current jobs to complete

2. **Scheduler stuck:**
   ```bash
   # Restart orchestrator
   sudo systemctl restart distribai-orchestrator
   ```

3. **Jobs in bad state:**
   ```bash
   # Cancel and recreate stuck jobs
   curl -X DELETE /admin/jobs/<job-id>
   ```

### Issue: High Node Failure Rate

**Symptoms:**
- Many nodes marked as "offline" or "degraded"
- Frequent reconnections in logs

**Diagnosis:**

```bash
# Check node heartbeat history
grep "heartbeat\|degraded\|offline" /var/log/distribai.log | tail -50

# Check network latency
ping <worker-host>

# Review worker logs on affected nodes
tail -100 /var/log/distribai-worker.log
```

**Solutions:**

1. **Network issues:**
   - Check for packet loss: `ping -c 100 <host>`
   - Verify stable internet connection
   - Consider regional orchestrator placement

2. **Clock skew:**
   ```bash
   # Sync clocks with NTP
   sudo ntpdate pool.ntp.org
   ```

3. **Resource exhaustion:**
   ```bash
   # Check worker resources
   free -h && df -h
   ```

### Issue: Slow Training Performance

**Symptoms:**
- Jobs take longer than expected
- Low GPU utilization on workers

**Diagnosis:**

```bash
# Check GPU utilization
nvidia-smi -l 1

# Monitor network throughput
iftop -i eth0

# Check S3 upload/download speeds
aws s3 cp test-file s3://bucket/ --profile distribai
```

**Solutions:**

1. **Small batch size:**
   - Increase `batch_size` in job parameters
   - Benchmark optimal size for hardware

2. **S3 latency:**
   - Use S3 bucket in same region as workers
   - Enable S3 Transfer Acceleration

3. **Gradient compression:**
   - Enable PowerSGD compression (reduces upload size)
   - Configured in `orchestrator_grpc.py`

### Issue: Byzantine Detection False Positives

**Symptoms:**
- Valid gradients marked as Byzantine
- Legitimate nodes penalized

**Diagnosis:**

```bash
# Check detection logs
grep -i "byzantine\|outlier\|krum" /var/log/distribai.log | tail -30

# Review gradient norms
# (requires custom logging in byzantine_detector.py)
```

**Solutions:**

1. **Adjust detection threshold:**
   ```python
   # In byzantine_detector.py
   BYZANTINE_THRESHOLD = 3.0  # Increase from 2.0
   ```

2. **Switch detection method:**
   ```python
   # In orchestrator_grpc.py initialization
   self.byzantine_detector = RobustAggregator(
       method=AggregationMethod.TRIMMED_MEAN  # Less aggressive than KRUM
   )
   ```

3. **Whitelist known good nodes:**
   ```python
   # Add to detection logic
   if node_id in TRUSTED_NODES:
       return False  # Skip Byzantine check
   ```

### Issue: Credit Ledger Verification Failures

**Symptoms:**
- Ledger verification returns `valid: false`
- Hash chain inconsistencies

**Diagnosis:**

```bash
# Run ledger verification
curl /admin/ledger/verify/0
curl /admin/ledger/verify/100
curl /admin/ledger/verify/500

# Check signing key consistency
echo $SIGNING_KEY | sha256sum
```

**Solutions:**

1. **Signing key mismatch:**
   - Ensure `SIGNING_KEY` is consistent across restarts
   - Set explicit key in `.env` file

2. **Database corruption:**
   ```bash
   # Restore from backup
   cp runtime/db/backup-$(date +%Y%m%d).db runtime/db/distribai.db
   ```

3. **Missing entries:**
   - Check for gaps in entry_id sequence
   - Rebuild ledger if necessary

### Issue: Rate Limiting Blocking Legitimate Requests

**Symptoms:**
- HTTP 429 errors
- Slow response times

**Diagnosis:**

```bash
# Check rate limiter logs
grep -i "rate.*limit\|429" /var/log/distribai.log | tail -20

# Test rate limit headers
curl -i /admin/health
# Look for: X-RateLimit-Remaining: 0
```

**Solutions:**

1. **Increase rate limits:**
   ```python
   # In rate_limiter.py
   DEFAULT_RATE = 20.0  # Increase from 10.0
   ```

2. **Whitelist internal IPs:**
   ```python
   # In rate_limiter.py
   WHITELISTED_IPS = ["127.0.0.1", "10.0.0.0/8"]
   ```

3. **Implement tiered limits:**
   - Higher limits for authenticated users
   - Lower limits for anonymous

### Issue: S3 Upload/Download Failures

**Symptoms:**
- Workers report "S3 error"
- Gradients not uploading
- Model weights not downloading

**Diagnosis:**

```bash
# Test S3 credentials
aws s3 ls s3://$S3_BUCKET_NAME --profile distribai

# Check bucket policy
aws s3api get-bucket-policy --bucket $S3_BUCKET_NAME

# Verify IAM permissions
aws iam get-user
```

**Solutions:**

1. **Expired credentials:**
   ```bash
   # Update AWS credentials
   aws configure --profile distribai
   ```

2. **Missing bucket permissions:**
   - Add `s3:GetObject`, `s3:PutObject` permissions
   - Verify bucket policy allows access

3. **Region mismatch:**
   ```bash
   # Set correct region
   export AWS_REGION=us-west-2
   ```

### Issue: Database Lock Errors

**Symptoms:**
- "database is locked" errors
- Slow query responses
- Timeouts

**Diagnosis:**

```bash
# Check for locks
lsof runtime/db/distribai.db

# Check database size
ls -lh runtime/db/distribai.db

# Check WAL mode
sqlite3 runtime/db/distribai.db "PRAGMA journal_mode;"
```

**Solutions:**

1. **Enable WAL mode:**
   ```bash
   sqlite3 runtime/db/distribai.db "PRAGMA journal_mode=WAL;"
   ```

2. **Increase timeout:**
   ```python
   # In db_manager.py
   conn.execute("PRAGMA busy_timeout = 30000")  # 30 seconds
   ```

3. **Archive old data:**
   ```bash
   # Backup and truncate old records
   sqlite3 runtime/db/distribai.db \
     "DELETE FROM credit_ledger WHERE created_at < $(date -d '30 days ago' +%s);"
   ```

### Issue: JWT Token Expiration

**Symptoms:**
- "Unauthorized" errors
- Nodes disconnected after 6 hours
- Token validation failures

**Diagnosis:**

```bash
# Decode JWT (header.payload.signature)
echo "<jwt-token>" | cut -d'.' -f2 | base64 -d | jq .
# Check "exp" field

# Verify JWT secret
echo $JWT_SECRET | sha256sum
```

**Solutions:**

1. **Extend token lifetime:**
   ```python
   # In orchestrator_grpc.py
   DEFAULT_NODE_JWT_TTL_SECONDS = 86400  # 24 hours
   ```

2. **Implement token refresh:**
   ```python
   # Add to worker daemon
   async def refresh_token(self):
       # Request new token before expiration
       pass
   ```

3. **Restart workers:**
   ```bash
   # Workers will get new tokens on reconnection
   sudo systemctl restart distribai-worker
   ```

## Debug Mode

### Enable Verbose Logging

```python
# In orchestrator_grpc.py
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Enable Worker Debug Mode

```bash
export DISTRIBAI_LOG_LEVEL=DEBUG
python -m worker.src.daemon.run
```

### Capture gRPC Traces

```bash
# Enable gRPC debugging
export GRPC_VERBOSITY=DEBUG
export GRPC_TRACE=all

python -m services_python.orchestrator_grpc 2>&1 | tee grpc-trace.log
```

## Emergency Procedures

### Complete System Restart

```bash
#!/bin/bash
# emergency-restart.sh

echo "Stopping all services..."
sudo systemctl stop distribai-orchestrator
sudo systemctl stop distribai-worker

echo "Clearing temporary files..."
rm -f runtime/db/*.lock
rm -f runtime/db/*.journal
rm -f runtime/db/*.wal

echo "Checking database integrity..."
sqlite3 runtime/db/distribai.db "PRAGMA integrity_check;"

echo "Starting services..."
sudo systemctl start distribai-orchestrator
sleep 5
sudo systemctl start distribai-worker

echo "Checking health..."
curl -sf http://localhost:8766/admin/health && echo "✓ System operational" || echo "✗ Health check failed"
```

### Data Recovery

```bash
#!/bin/bash
# data-recovery.sh

BACKUP_DIR="/backup/distribai"
DATE=$(date +%Y%m%d_%H%M%S)

# 1. Stop services
sudo systemctl stop distribai-orchestrator

# 2. Backup current (potentially corrupted) state
cp -r runtime/db "runtime/db-corrupted-$DATE"

# 3. Find latest good backup
LATEST_BACKUP=$(ls -t $BACKUP_DIR/distribai-*.db | head -1)
echo "Restoring from: $LATEST_BACKUP"

# 4. Restore
cp "$LATEST_BACKUP" runtime/db/distribai.db

# 5. Start services
sudo systemctl start distribai-orchestrator

# 6. Verify
sleep 5
curl http://localhost:8766/admin/health
```

## Contact & Escalation

For support, bug reports, and feature requests, use the [DistribAI GitHub Issues](https://github.com/naxium-oss/DistribAI/issues) tracker. Include the affected component, relevant logs, reproduction steps, and environment details.

| Severity | Response Time | Action |
|----------|---------------|--------|
| Critical (system down) | 15 min | Page on-call, begin emergency procedures |
| High (major impact) | 1 hour | Notify team lead, document in incident log |
| Medium (partial impact) | 4 hours | Create ticket, monitor |
| Low (minor issue) | 24 hours | Add to backlog |

## Useful Commands Reference

```bash
# Process monitoring
watch -n 1 'ps aux | grep distribai'

# Network monitoring
watch -n 1 'netstat -tlnp | grep -E "50051|8766"'

# Resource monitoring
watch -n 1 'nvidia-smi'

# Log monitoring
tail -f /var/log/distribai.log | grep -E "ERROR|WARNING|CRITICAL"

# Database queries
sqlite3 runtime/db/distribai.db "SELECT * FROM active_nodes WHERE status = 'offline';"

# API testing
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8766/admin/health
```
