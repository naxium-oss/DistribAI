# Monitoring & Alerting Runbook

## Overview

This runbook covers monitoring setup, metric collection, and alerting configuration for DistribAI.

## Key Metrics

### Orchestrator Metrics

| Metric | Type | Threshold | Alert |
|--------|------|-----------|-------|
| Node heartbeats/sec | Counter | < 0.5 | Warning |
| Active nodes | Gauge | < 5 | Critical |
| Job queue depth | Gauge | > 100 | Warning |
| Job completion rate | Rate | < 10/hr | Warning |
| Failed jobs | Counter | > 5% | Critical |
| API latency (p99) | Histogram | > 500ms | Warning |
| Error rate | Rate | > 1% | Critical |
| Byzantine nodes detected | Counter | > 0 | Info |
| Credit transactions/sec | Rate | N/A | Info |

### Worker Metrics

| Metric | Type | Threshold | Alert |
|--------|------|-----------|-------|
| GPU utilization | Gauge | < 20% for 1hr | Warning |
| VRAM usage | Gauge | > 95% | Warning |
| Training throughput | Rate | < expected | Warning |
| Connection status | State | disconnected | Critical |
| Task completion time | Histogram | > deadline | Critical |
| Gradient upload time | Histogram | > 30s | Warning |

## Metric Collection

### Via Admin API

```bash
# Get current metrics
curl http://localhost:8766/admin/health
curl http://localhost:8766/admin/nodes
curl http://localhost:8766/admin/jobs
```

### Desktop Dashboard

The supported monitoring surface is the desktop dashboard plus the admin API.
Start it with:

```bash
python -m services_python.server_gui
```

The dashboard reads the same `/admin/*` endpoints shown above.

## Health Checks

### Basic Health Check

```bash
#!/bin/bash
# Basic health check script

ORCHESTRATOR_URL="http://localhost:8766"
ALERT_WEBHOOK="https://hooks.slack.com/your/webhook"

# Check orchestrator health
if ! curl -sf "${ORCHESTRATOR_URL}/admin/health" > /dev/null; then
    echo "$(date): CRITICAL - Orchestrator not responding"
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🚨 DistribAI orchestrator is DOWN"}' \
        "$ALERT_WEBHOOK"
    exit 1
fi

# Check for degraded nodes
NODES=$(curl -sf "${ORCHESTRATOR_URL}/admin/nodes")
DEGRADED=$(echo "$NODES" | jq '[.nodes[] | select(.status == "degraded")] | length')
if [ "$DEGRADED" -gt 5 ]; then
    echo "$(date): WARNING - $DEGRADED nodes degraded"
fi

# Check queue depth
QUEUE=$(curl -sf "${ORCHESTRATOR_URL}/admin/jobs" | jq '.queue_depth')
if [ "$QUEUE" -gt 100 ]; then
    echo "$(date): WARNING - Queue depth: $QUEUE"
fi

echo "$(date): Health check passed"
```

### Advanced Health Check (Python)

```python
#!/usr/bin/env python3
"""Advanced health monitoring script."""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta

class HealthMonitor:
    def __init__(self, orchestrator_url: str):
        self.url = orchestrator_url
        self.alerts = []
    
    async def check_health(self) -> dict:
        """Run all health checks."""
        async with aiohttp.ClientSession() as session:
            checks = {
                'orchestrator': await self._check_orchestrator(session),
                'nodes': await self._check_nodes(session),
                'jobs': await self._check_jobs(session),
                'credits': await self._check_credits(session),
            }
            
            # Overall status
            critical = any(c['status'] == 'critical' for c in checks.values())
            warning = any(c['status'] == 'warning' for c in checks.values())
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'overall': 'critical' if critical else 'warning' if warning else 'healthy',
                'checks': checks
            }
    
    async def _check_orchestrator(self, session) -> dict:
        """Check orchestrator health."""
        try:
            async with session.get(f"{self.url}/admin/health", timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        'status': 'healthy',
                        'details': data
                    }
                return {
                    'status': 'critical',
                    'details': f'HTTP {resp.status}'
                }
        except Exception as e:
            return {
                'status': 'critical',
                'details': str(e)
            }
    
    async def _check_nodes(self, session) -> dict:
        """Check node health."""
        try:
            async with session.get(f"{self.url}/admin/nodes", timeout=5) as resp:
                data = await resp.json()
                nodes = data.get('nodes', [])
                
                total = len(nodes)
                working = sum(1 for n in nodes if n['status'] == 'working')
                idle = sum(1 for n in nodes if n['status'] == 'idle')
                degraded = sum(1 for n in nodes if n['status'] == 'degraded')
                offline = sum(1 for n in nodes if n['status'] == 'offline')
                
                # Determine status
                if total == 0:
                    status = 'warning'
                elif offline > total * 0.5:
                    status = 'critical'
                elif degraded > total * 0.3:
                    status = 'warning'
                else:
                    status = 'healthy'
                
                return {
                    'status': status,
                    'details': {
                        'total': total,
                        'working': working,
                        'idle': idle,
                        'degraded': degraded,
                        'offline': offline
                    }
                }
        except Exception as e:
            return {
                'status': 'warning',
                'details': str(e)
            }
    
    async def _check_jobs(self, session) -> dict:
        """Check job health."""
        try:
            async with session.get(f"{self.url}/admin/jobs", timeout=5) as resp:
                data = await resp.json()
                queue_depth = data.get('queue_depth', 0)
                
                # Check for stuck jobs
                jobs = data.get('jobs', [])
                stuck = [
                    j for j in jobs
                    if j['status'] == 'running' and 
                    self._is_stuck(j)
                ]
                
                status = 'healthy'
                if queue_depth > 100:
                    status = 'warning'
                if stuck:
                    status = 'critical'
                
                return {
                    'status': status,
                    'details': {
                        'queue_depth': queue_depth,
                        'stuck_jobs': len(stuck)
                    }
                }
        except Exception as e:
            return {
                'status': 'warning',
                'details': str(e)
            }
    
    def _is_stuck(self, job: dict) -> bool:
        """Check if job appears stuck."""
        updated = job.get('updated_at', 0)
        if updated:
            last_update = datetime.fromtimestamp(updated)
            return datetime.utcnow() - last_update > timedelta(hours=1)
        return False
    
    async def _check_credits(self, session) -> dict:
        """Check credit system health."""
        # This is a simple check - could be expanded
        return {
            'status': 'healthy',
            'details': 'Credit system operational'
        }

async def main():
    monitor = HealthMonitor("http://localhost:8766")
    result = await monitor.check_health()
    print(json.dumps(result, indent=2))
    
    # Exit with error code if not healthy
    if result['overall'] != 'healthy':
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Alerting Configuration

### PagerDuty Integration

```python
import requests

class PagerDutyAlerter:
    def __init__(self, routing_key: str):
        self.routing_key = routing_key
        self.url = "https://events.pagerduty.com/v2/enqueue"
    
    def alert(self, summary: str, severity: str, details: dict = None):
        """Send alert to PagerDuty."""
        payload = {
            "routing_key": self.routing_key,
            "event_action": "trigger",
            "dedup_key": summary[:255],
            "payload": {
                "summary": summary,
                "severity": severity,  # critical, error, warning, info
                "source": "distribai",
                "custom_details": details or {}
            }
        }
        
        requests.post(self.url, json=payload)
```

### Slack Integration

```python
import requests

class SlackAlerter:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def alert(self, message: str, level: str = "info"):
        """Send alert to Slack."""
        colors = {
            "info": "#36a64f",
            "warning": "#ff9900",
            "critical": "#ff0000"
        }
        
        payload = {
            "attachments": [{
                "color": colors.get(level, "#36a64f"),
                "text": message,
                "footer": "DistribAI Monitoring",
                "ts": int(datetime.utcnow().timestamp())
            }]
        }
        
        requests.post(self.webhook_url, json=payload)
```

### Email Alerts

```python
import smtplib
from email.mime.text import MIMEText

class EmailAlerter:
    def __init__(self, smtp_host: str, smtp_port: int, from_addr: str, to_addrs: list):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_addr = from_addr
        self.to_addrs = to_addrs
    
    def alert(self, subject: str, body: str):
        """Send email alert."""
        msg = MIMEText(body)
        msg['Subject'] = f"[DistribAI] {subject}"
        msg['From'] = self.from_addr
        msg['To'] = ', '.join(self.to_addrs)
        
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.send_message(msg)
```

## Dashboard Metrics

### Real-Time Status Board

```html
<!-- Simple status dashboard -->
<!DOCTYPE html>
<html>
<head>
    <title>DistribAI Status</title>
    <script>
        async function updateStatus() {
            const response = await fetch('/admin/health');
            const health = await response.json();
            document.getElementById('status').textContent = health.ok ? '✓ Healthy' : '✗ Down';
            
            const nodesResponse = await fetch('/admin/nodes');
            const nodes = await nodesResponse.json();
            document.getElementById('nodes').textContent = nodes.nodes.length;
            
            const jobsResponse = await fetch('/admin/jobs');
            const jobs = await jobsResponse.json();
            document.getElementById('queue').textContent = jobs.queue_depth;
        }
        
        setInterval(updateStatus, 5000);
        updateStatus();
    </script>
</head>
<body>
    <h1>DistribAI Status</h1>
    <div>Status: <span id="status">Loading...</span></div>
    <div>Active Nodes: <span id="nodes">-</span></div>
    <div>Queue Depth: <span id="queue">-</span></div>
</body>
</html>
```

## Log Analysis

### Common Log Patterns

```bash
# Check for errors
grep -i "error" /var/log/distribai.log | tail -20

# Check for failed jobs
grep "failed" /var/log/distribai.log | tail -20

# Check node disconnections
grep "disconnected\|reconnecting" /var/log/distribai.log | tail -20

# Check Byzantine detections
grep "byzantine\|outlier" /var/log/distribai.log | tail -20
```

### Log Aggregation with ELK

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  paths:
    - /var/log/distribai.log
  fields:
    service: distribai
    component: orchestrator

output.logstash:
  hosts: ["localhost:5044"]
```

## Troubleshooting

### No Heartbeats from Nodes

1. Check network connectivity: `ping orchestrator-host`
2. Verify gRPC port is open: `telnet orchestrator-host 50051`
3. Check JWT token validity
4. Review worker logs for errors

### High Queue Depth

1. Check number of active workers
2. Verify workers can receive tasks (not all busy)
3. Check for stuck jobs: `curl /admin/jobs | jq '.jobs[] | select(.status == "running")'`
4. Review scheduler logs

### Slow Job Completion

1. Check worker GPU utilization
2. Verify S3 upload/download speeds
3. Review task decomposition settings
4. Check for network bottlenecks

### Credit Ledger Discrepancies

1. Run ledger verification: `curl /admin/ledger/verify/0`
2. Check for duplicate transactions
3. Review transaction logs
4. Verify signing key consistency
