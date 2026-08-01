# RFC 010: Cloud Spot Instance Integration

**Status:** Completed  
**Date:** 2026-04-21  
**Research Lead:** DistribAI Infrastructure Team

---

## Summary

Research on integrating AWS/GCP spot instances as DistribAI surge capacity. Analyzes preemption handling, checkpoint strategies, and cost-benefit at various network scales.

## Spot Instance Fundamentals

### AWS EC2 Spot
- **Discount**: Up to 90% vs on-demand
- **Preemption**: 2-minute warning before termination
- **Use cases**: Fault-tolerant, checkpointed workloads

### GCP Preemptible
- **Discount**: ~80% vs standard
- **Preemption**: 30-second warning
- **Limit**: Max 24 hours runtime

## Preemption Handling Strategy

### 2-Minute Checkpoint Window (AWS)
```python
# EventBridge captures preemption notice
# Lambda triggers immediate checkpoint
# Graceful shutdown within 90 seconds
```

**Implementation:**
1. Poll instance metadata service every 5 seconds
2. On termination notice: immediately checkpoint model state
3. Upload checkpoint to S3 within 30 seconds
4. Signal orchestrator of preemption

### Cost-Benefit Analysis

| Network Size | Spot Useful? | Primary Benefit |
|--------------|--------------|-----------------|
| < 100 nodes | No | Management overhead exceeds savings |
| 100-500 nodes | Marginal | ~15% cost reduction |
| 500-2000 nodes | Yes | ~30% cost reduction on surge |
| 2000+ nodes | Strong Yes | ~50% cost reduction |

**Break-even point**: ~400 active nodes

## DistribAI Integration Plan

### Phase 4: Optional Spot Workers

**Configuration:**
```toml
[cloud]
spot_enabled = true
provider = "aws"  # or "gcp"
max_spot_nodes = 100
instance_types = ["g4dn.xlarge", "p3.2xlarge"]
```

**Orchestrator Changes:**
1. Tag spot workers in node registry
2. Assign only checkpoint-resumable jobs to spots
3. Monitor preemption events via webhook
4. Auto-requeue preempted tasks

**Checkpoint Frequency:**
- Standard nodes: Every 100 steps
- Spot nodes: Every 10 steps (higher overhead, lower risk)

### Preemption Response Flow
```
Spot Instance
    ↓ (preemption notice)
EventBridge Rule
    ↓
Lambda Function
    ↓ (trigger checkpoint)
Worker Daemon
    ↓ (upload within 90s)
S3 Checkpoint
    ↓
Orchestrator Requeue
```

## References

1. AWS "Best practices for handling EC2 Spot Instance interruptions"
2. AWS "Checkpointing HPC applications using the Spot Instance two-minute notification"
3. GCP "Preemptible VM instances" documentation

---
*End of RFC 010*
