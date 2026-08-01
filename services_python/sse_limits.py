"""Caps on concurrent connections and outbound bytes for admin SSE streams."""

from __future__ import annotations

import asyncio
import os
import time
from collections import defaultdict


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def max_sse_connections() -> int:
    return _positive_int_from_env("ADMIN_SSE_MAX_CONNECTIONS", 32)


def max_sse_connections_per_ip() -> int:
    return _positive_int_from_env("ADMIN_SSE_MAX_PER_IP", 8)


def max_sse_bytes_per_sec() -> int:
    """Per-connection outbound byte rate for /admin/stream (0 disables limiting)."""
    return _positive_int_from_env("ADMIN_SSE_MAX_BYTES_PER_SEC", 65536)


class SseByteBudget:
    """Token-bucket throttle for bytes leaving a single SSE connection."""

    def __init__(self, bytes_per_sec: int | None = None) -> None:
        rate = bytes_per_sec if bytes_per_sec is not None else max_sse_bytes_per_sec()
        self._rate = max(0, rate)
        self._allowance = float(self._rate)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def wait_for(self, nbytes: int) -> None:
        if self._rate <= 0 or nbytes <= 0:
            return
        remaining = nbytes
        async with self._lock:
            while remaining > 0:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                self._allowance = min(
                    float(self._rate),
                    self._allowance + elapsed * self._rate,
                )
                if self._allowance >= remaining:
                    self._allowance -= remaining
                    return
                remaining -= int(self._allowance)
                self._allowance = 0.0
                await asyncio.sleep(remaining / self._rate)


class AdminSseLimiter:
    """Counts open /admin/stream sessions to blunt long-lived connection abuse."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._total = 0
        self._per_ip: dict[str, int] = defaultdict(int)

    async def try_acquire(self, client_ip: str) -> bool:
        async with self._lock:
            if self._total >= max_sse_connections():
                return False
            ip = client_ip or "unknown"
            if self._per_ip[ip] >= max_sse_connections_per_ip():
                return False
            self._total += 1
            self._per_ip[ip] += 1
            return True

    async def release(self, client_ip: str) -> None:
        async with self._lock:
            ip = client_ip or "unknown"
            count = self._per_ip.get(ip, 0)
            if count <= 0:
                return
            self._total = max(0, self._total - 1)
            if count == 1:
                del self._per_ip[ip]
            else:
                self._per_ip[ip] = count - 1


_admin_sse_limiter = AdminSseLimiter()


def admin_sse_limiter() -> AdminSseLimiter:
    return _admin_sse_limiter
