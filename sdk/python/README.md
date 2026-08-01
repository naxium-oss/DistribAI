# DistribAI Python SDK

<p style="max-width:40rem;margin:0 auto 1rem;line-height:1.55;text-align:center;">
  <a href="../../README.md"><strong>README</strong></a> &nbsp;|&nbsp; <span>(monorepo)</span><br/>
  <a href="../../TODO.md"><strong>TODO</strong></a> &nbsp;|&nbsp; <span>(backlog)</span><br/>
  <a href="../../AGENTS.md"><strong>AGENTS</strong></a> &nbsp;|&nbsp; <span>(rules)</span><br/>
  <a href="../../docs/README.md"><strong>Documentation index</strong></a> &nbsp;|&nbsp; <span>(guides)</span><br/>
  <a href="../../docs/runbooks/deployment.md"><strong>Deploy runbook</strong></a> &nbsp;|&nbsp; <span>(ops)</span>
</p>

A clean Python SDK surface for programmatic interaction with orchestrator-facing HTTP APIs documented under [`docs/api/`](../docs/api/).

## Installation

```bash
pip install distribai
```

## Quick Start

```python
import distribai
import asyncio

async def main():
    # Create a client
    client = distribai.Client(api_key="your-api-key")
    
    # Submit a training job
    job = await client.jobs.submit(
        model_name="small",
        dataset="s3://datasets/mydata.jsonl",
        steps=1000,
        batch_size=32
    )
    
    print(f"Job submitted: {job.id}")
    print(f"Queue position: {job.queue_position}")
    
    # Wait for completion
    await job.wait_for_completion()
    print("Job completed!")
    
    # Check credits earned
    balance = await client.credits.balance()
    print(f"Available credits: {balance.confirmed}")

asyncio.run(main())
```

## Features

- **Async/Await Support**: Fully async API for efficient concurrent operations
- **Type Hints**: Complete type annotations for better IDE support
- **Resource Management**: Async context manager support for proper cleanup
- **Error Handling**: Specific exceptions for different error types
- **Pagination**: Automatic handling of paginated responses

## API Reference

### Client

```python
from distribai import Client

client = Client(
    api_key="cg_live_...",
    base_url="http://127.0.0.1:8766",  # Optional — your orchestrator admin URL
    timeout=30.0,  # Optional
)

# Or use as async context manager
async with Client(api_key="...") as client:
    # Use client here
    pass
```

### Jobs

```python
# Submit a job
job = await client.jobs.submit(
    model_name="small",  # or "medium", "large", "custom"
    dataset="s3://datasets/train.jsonl",
    steps=5000,
    batch_size=32,
    priority=5,  # 0-10
    hparams={"lr": 1e-4, "warmup_steps": 100}
)

# Get job status
job = await client.jobs.get(job.id)
print(f"Status: {job.status}")
print(f"Progress: {job.progress_pct}%")

# List jobs
jobs = await client.jobs.list(status=JobStatus.RUNNING, limit=10)

# Wait for completion
await job.wait_for_completion(poll_interval=5.0)

# Cancel job
await job.cancel()
```

### Credits

```python
# Check balance
balance = await client.credits.balance()
print(f"Confirmed: {balance.confirmed}")
print(f"Pending: {balance.pending}")
print(f"Lifetime earned: {balance.lifetime_earned}")

# View transaction history
transactions = await client.credits.history(limit=20)
for tx in transactions:
    print(f"{tx.timestamp}: {tx.type} {tx.amount} credits")

# Transfer credits
await client.credits.transfer(
    to_node_id="node_abc123",
    amount=100.0,
    reason="Payment for dataset preparation"
)

# Check earning rate
rate = await client.credits.estimated_earning_rate()
print(f"Estimated: {rate['credits_per_hour']} credits/hour")
```

### Voting

```python
# Cast a vote
result = await client.votes.cast(
    job_id="job_abc123",
    credits=50  # Spend 50 credits
)
print(f"Job new position: #{result['job_new_queue_position']}")

# View queue
queue = await client.votes.queue()
for job in queue[:10]:
    print(f"{job.queue_position}. {job.title} ({job.total_votes} credits)")

# Get vote tally for specific job
tally = await client.votes.get_vote_tally("job_abc123")
print(f"Total votes: {tally['total_credits']}")
```

### Nodes

```python
# Register a new node
registration = await client.nodes.register(
    invite_code="optional-invite-code",
    public_key="optional-public-key"
)
node_id = registration["node_id"]
jwt_token = registration["jwt"]

# List nodes
nodes = await client.nodes.list(status=NodeStatus.IDLE)
for node in nodes:
    print(f"{node.id}: {node.status.value} (GPU: {node.hardware.gpu_model})")

# Get node details
node = await client.nodes.get("node_abc123")
print(f"Reliability: {node.reliability_score}")
print(f"Credits earned: {node.credits_earned}")

# Get network stats
stats = await client.nodes.get_stats()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Online: {stats['online_nodes']}")
```

## Error Handling

```python
from distribai import (
    DistribAIError,
    AuthenticationError,
    RateLimitError,
    JobNotFoundError,
    InsufficientCreditsError,
)

try:
    job = await client.jobs.get("nonexistent-job")
except JobNotFoundError as e:
    print(f"Job not found: {e.job_id}")
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after}s")
except InsufficientCreditsError as e:
    print(f"Need {e.required} credits, have {e.available}")
except DistribAIError as e:
    print(f"API error: {e.message} (code: {e.error_code})")
```

## License

Apache License 2.0 - see the repository LICENSE file for details.
