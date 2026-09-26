"""Executor fails loudly when weight download fails."""

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
async def test_load_weights_raises_when_download_fails():
    async def noop(*_args, **_kwargs):
        return None

    executor = JobExecutor("test-node", noop, noop)

    class _FakeS3:
        async def download_file(self, *_args, **_kwargs):
            return False

    executor.s3 = _FakeS3()

    import torch.nn as nn

    model = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Failed to download weights"):
        await executor._load_weights(model, "task-1", "https://127.0.0.1/weights.pt")
