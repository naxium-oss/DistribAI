# Chaos Engineering Guide

**Purpose:** Systematically test DistribAI resilience through controlled failure injection.

---

## Philosophy

> "The best way to avoid failure is to fail constantly." — Netflix Chaos Engineering Team

We inject failures to:
1. Identify weaknesses before they cause incidents
2. Validate monitoring and alerting
3. Train incident response
4. Build confidence in system resilience

---

## Chaos Experiments

### Experiment 1: Node Mass Failure

**Objective:** Verify graceful handling of sudden node loss.

**Setup:**
```bash
# Target: 20% of active nodes
TARGET_COUNT=$(curl -s http://localhost:8766/admin/nodes | jq '.nodes | length * 0.2 | floor')
echo "Will disconnect $TARGET_COUNT nodes"
```

**Execution:**
```python
# chaos_node_failure.py
import asyncio
import random
import requests

async def kill_random_nodes(orchestrator_url: str, percentage: float = 0.2):
    """Simulate random node failures."""
    resp = requests.get(f"{orchestrator_url}/admin/nodes")
    nodes = resp.json()["nodes"]
    target_count = int(len(nodes) * percentage)
    victims = random.sample(nodes, target_count)
    
    for node in victims:
        # Send disconnect signal
        requests.post(
            f"{orchestrator_url}/admin/nodes/{node['node_id']}/chaos",
            json={"action": "force_disconnect", "duration_sec": 300}
        )
        print(f"Disconnected {node['node_id']}")
        await asyncio.sleep(1)  # Stagger disconnections

# Run
asyncio.run(kill_random_nodes("http://localhost:8766", 0.2))
```

**Expected Behavior:**
- Tasks from failed nodes re-queued within 30 seconds
- No data loss
- Training continues with remaining nodes
- Alerts fired if success rate drops below 80%

**Success Criteria:**
```bash
# Check job success rate after 5 minutes
python -c "
import requests
import sys
jobs = requests.get('http://localhost:8766/admin/jobs').json()['jobs']
success_rate = sum(1 for j in jobs if j['status'] == 'success') / len(jobs)
print(f'Success rate: {success_rate:.1%}')
sys.exit(0 if success_rate > 0.8 else 1)
"
```

---

### Experiment 2: Byzantine Node Injection

**Objective:** Test Byzantine detection under controlled attack.

**Setup:**
```python
# chaos_byzantine.py
import torch
import numpy as np
from worker.src.distribai_proto import distribai_pb2

class ByzantineWorker:
    """Worker that sends malicious gradients."""
    
    def __init__(self, attack_type: str = "sign_flipping"):
        self.attack_type = attack_type
        
    def generate_malicious_gradient(self, shape: tuple) -> torch.Tensor:
        if self.attack_type == "sign_flipping":
            # Send negated gradients
            return -torch.randn(shape) * 10
        elif self.attack_type == "random_noise":
            # Send random values
            return torch.randn(shape) * 1000
        elif self.attack_type == "targeted":
            # Send gradients that push model toward wrong direction
            return torch.ones(shape) * 50
        else:
            raise ValueError(f"Unknown attack: {self.attack_type}")
```

**Execution:**
```bash
# Inject 3 Byzantine nodes among 20 honest nodes
python chaos_byzantine.py --count 3 --attack sign_flipping --duration 600
```

**Expected Detection:**
- Byzantine nodes flagged within 2 aggregation rounds
- Reputation scores drop for malicious nodes
- Aggregation method may switch to more robust variant

**Verify:**
```bash
curl http://localhost:8766/admin/byzantine-stats | jq '.flagged_nodes'
```

---

### Experiment 3: Credit System Stress Test

**Objective:** Verify ledger integrity under high transaction volume.

**Execution:**
```python
# chaos_credit_flood.py
import asyncio
import random
from services_python.credit_transfers import CreditTransferManager

async def flood_transactions(db, duration_sec: int = 60, tps: int = 100):
    """Generate high-volume credit transactions."""
    manager = CreditTransferManager(db)
    start = time.time()
    count = 0
    
    while time.time() - start < duration_sec:
        tasks = []
        for _ in range(tps):
            sender = f"node-{random.randint(1, 100)}"
            receiver = f"node-{random.randint(1, 100)}"
            amount = random.uniform(1, 100)
            tasks.append(manager.transfer(sender, receiver, amount))
        
        await asyncio.gather(*tasks, return_exceptions=True)
        count += tps
        await asyncio.sleep(1)
    
    print(f"Completed {count} transactions")
    
    # Verify ledger integrity
    from worker.src.daemon.credit_ledger import CreditLedger
    ledger = CreditLedger()
    assert ledger.verify_chain(0, ledger.head.size), "Ledger corruption!"
    print("✅ Ledger integrity verified")
```

---

### Experiment 4: Network Partition

**Objective:** Test split-brain prevention.

**Setup:**
```bash
# Using iptables to partition network
# Split nodes into two groups that can't communicate

# Group A: Nodes 1-25
# Group B: Nodes 26-50

sudo iptables -A INPUT -s 10.0.1.0/25 -d 10.0.1.128/25 -j DROP
sudo iptables -A OUTPUT -s 10.0.1.0/25 -d 10.0.1.128/25 -j DROP
```

**Expected Behavior:**
- Each partition continues with available nodes
- No double-spending of credits
- Quorum-based voting blocked until partition heals

**Recovery:**
```bash
# Heal partition
sudo iptables -F

# Verify convergence
watch -n 5 'curl -s http://localhost:8766/admin/nodes | jq ".nodes | length"'
```

---

### Experiment 5: Database Corruption

**Objective:** Test backup/recovery procedures.

**Execution:**
```bash
# 1. Note current state
sqlite3 /path/to/distribai.db "SELECT COUNT(*) FROM jobs;" > /tmp/pre_chaos_count.txt

# 2. Simulate corruption (TEST ENVIRONMENT ONLY!)
sqlite3 /path/to/distribai.db "DELETE FROM tasks WHERE RANDOM() % 10 = 0;"

# 3. Detect corruption
python scripts/verify_database_integrity.py
# Expected: Reports missing task records

# 4. Restore from backup
systemctl stop distribai-orchestrator
cp /backups/distribai-$(date -d '1 hour ago' +%Y%m%d-%H).db /path/to/distribai.db
systemctl start distribai-orchestrator

# 5. Verify restoration
sqlite3 /path/to/distribai.db "SELECT COUNT(*) FROM jobs;" > /tmp/post_restore_count.txt
diff /tmp/pre_chaos_count.txt /tmp/post_restore_count.txt && echo "✅ Restore successful"
```

---

### Experiment 6: S3 Storage Failure

**Objective:** Test degraded mode operation.

**Execution:**
```bash
# Block S3 access (simulates outage)
sudo iptables -A OUTPUT -d s3.amazonaws.com -j DROP

# Wait 60 seconds for detection
sleep 60

# System should:
# - Switch to local storage fallback
# - Queue checkpoints locally
# - Alert on-call engineer

# Verify degraded mode
curl http://localhost:8766/admin/health | jq '.storage_mode'
# Expected: "local_fallback"

# Restore S3
sudo iptables -D OUTPUT -d s3.amazonaws.com -j DROP

# Verify sync back to S3
watch -n 10 'aws s3 ls s3://$S3_BUCKET_NAME/checkpoints/ | wc -l'
```

---

## Chaos Automation

### Scheduled Chaos (GameDays)

**Monthly GameDay Schedule:**

```yaml
# chaos-schedule.yaml
experiments:
  - name: "node_failure_small"
    schedule: "0 14 * * 1"  # Mondays 2pm
    params:
      percentage: 0.1
      duration_sec: 300
    
  - name: "byzantine_injection"
    schedule: "0 10 * * 3"  # Wednesdays 10am
    params:
      count: 2
      attack_types: ["sign_flipping", "random_noise"]
    
  - name: "credit_flood"
    schedule: "0 16 15 * *"  # 15th of month, 4pm
    params:
      tps: 500
      duration_sec: 120
    
  - name: "full_drill"
    schedule: "0 9 1 * *"  # 1st of month, full drill
    params:
      scenario: "orchestrator_crash"
      expected_recovery_sec: 300
```

**Run with:**
```bash
python -m chaos.engine run --schedule chaos-schedule.yaml --notify #incident-response
```

---

## Safety Guidelines

### Golden Rules

1. **Never in Production Without Approval**
   - SEV1 requires VP sign-off
   - Always have rollback ready

2. **Blast Radius Limits**
   - Max 20% nodes affected
   - Max 1 hour duration
   - Never touch credit ledger in production

3. **Monitoring Required**
   - Incident commander must be online
   - Auto-rollback if success rate < 50%

4. **Document Everything**
   - Pre-experiment state
   - Observed behavior
   - Deviations from expected

### Abort Conditions

Immediately stop if:
- Credit ledger reports inconsistency
- >30% of jobs failing
- Data loss detected
- Manual abort signal received

**Abort Command:**
```bash
# Emergency stop all chaos experiments
touch /tmp/CHAOS_ABORT
curl -X POST http://localhost:8766/admin/chaos/abort
```

---

## Measuring Resilience

### Metrics to Track

| Metric | Baseline | Target Under Chaos |
|--------|----------|-------------------|
| Job success rate | 95% | >80% |
| Node recovery time | N/A | <60 sec |
| Credit ledger latency | <10ms | <100ms |
| False positive rate (Byzantine) | <1% | <5% |
| Data loss | 0 | 0 (hard requirement) |

### Resilience Score

```python
def calculate_resilience_score(experiment_results: dict) -> float:
    """
    Calculate overall resilience score from experiment results.
    Score 0-100, higher is better.
    """
    weights = {
        'success_rate': 0.3,
        'recovery_time': 0.25,
        'data_integrity': 0.25,
        'detection_accuracy': 0.2
    }
    
    scores = {
        'success_rate': min(100, experiment_results['success_rate'] * 100),
        'recovery_time': max(0, 100 - (experiment_results['recovery_sec'] / 3)),
        'data_integrity': 100 if experiment_results['data_loss'] == 0 else 0,
        'detection_accuracy': experiment_results['true_positive_rate'] * 100
    }
    
    return sum(scores[k] * weights[k] for k in weights)
```

---

## Chaos Engineering Checklist

Before running any experiment:

- [ ] Staging environment confirmed
- [ ] Rollback procedure tested
- [ ] Incident commander assigned
- [ ] Monitoring dashboards open
- [ ] Abort mechanism verified
- [ ] Blast radius calculated
- [ ] Team notified in #incident-response
- [ ] Experiment documented in runbook

---

**Next GameDay:** [Date]  
**Chaos Engineering Owner:** Infrastructure Team
