"""Tests for admin SSE connection and bandwidth limits."""

from __future__ import annotations

import time

import pytest

from services_python.sse_limits import (
    AdminSseLimiter,
    SseByteBudget,
    max_sse_bytes_per_sec,
    max_sse_connections,
    max_sse_connections_per_ip,
)


@pytest.mark.asyncio
async def test_sse_limiter_enforces_total_cap(monkeypatch):
    monkeypatch.setenv("ADMIN_SSE_MAX_CONNECTIONS", "2")
    monkeypatch.setenv("ADMIN_SSE_MAX_PER_IP", "10")
    limiter = AdminSseLimiter()
    assert await limiter.try_acquire("1.2.3.4") is True
    assert await limiter.try_acquire("5.6.7.8") is True
    assert await limiter.try_acquire("1.2.3.4") is False
    await limiter.release("1.2.3.4")
    assert await limiter.try_acquire("1.2.3.4") is True


@pytest.mark.asyncio
async def test_sse_limiter_enforces_per_ip_cap(monkeypatch):
    monkeypatch.setenv("ADMIN_SSE_MAX_CONNECTIONS", "100")
    monkeypatch.setenv("ADMIN_SSE_MAX_PER_IP", "1")
    limiter = AdminSseLimiter()
    assert await limiter.try_acquire("10.0.0.1") is True
    assert await limiter.try_acquire("10.0.0.1") is False


def test_sse_limit_defaults():
    assert max_sse_connections() >= 1
    assert max_sse_connections_per_ip() >= 1
    assert max_sse_bytes_per_sec() >= 1


@pytest.mark.asyncio
async def test_sse_byte_budget_throttles_large_burst():
    budget = SseByteBudget(bytes_per_sec=2000)
    t0 = time.monotonic()
    await budget.wait_for(4000)
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.8
