"""Worker executor batch blob URL policy (parity with gradient allowlist)."""

from __future__ import annotations

import pytest

try:
    from worker.src.daemon.executor import JobExecutor

    HAS_EXECUTOR = True
except ImportError:
    HAS_EXECUTOR = False
    JobExecutor = None


@pytest.mark.skipif(not HAS_EXECUTOR, reason="executor not available")
@pytest.mark.asyncio
async def test_load_batch_source_blocks_disallowed_https(monkeypatch):
    monkeypatch.setenv("ALLOWED_BLOB_HOSTS", "127.0.0.1,localhost")

    async def noop(*_args, **_kwargs):
        return None

    executor = JobExecutor("test-node", noop, noop)

    with pytest.raises(ValueError, match="Unauthorized batch blob URL"):
        await executor._load_batch_source("task-1", "https://evil.example.com/batch.json")
