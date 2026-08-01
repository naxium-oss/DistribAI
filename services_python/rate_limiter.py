"""
Rate Limiter for DistribAI API
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""

    rate: float
    capacity: float
    tokens: float = field(default=0.0)
    last_update: float = field(default_factory=time.time)

    def consume(self, tokens: float = 1.0) -> bool:
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_wait_time(self, tokens: float = 1.0) -> float:
        now = time.time()
        elapsed = now - self.last_update
        current_tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        if current_tokens >= tokens:
            return 0.0
        needed = tokens - current_tokens
        return needed / self.rate


class RateLimiter:
    """Token bucket rate limiter"""

    DEFAULT_RATE = 10.0
    DEFAULT_CAPACITY = 20.0
    DEFAULT_BLOCK_DURATION = 60.0
    MAX_BUCKETS = int(os.getenv("RATE_LIMIT_MAX_BUCKETS", "10000"))
    MAX_BLOCKED_CLIENTS = int(os.getenv("RATE_LIMIT_MAX_BLOCKED", "5000"))
    MAX_VIOLATIONS_TRACKED = int(os.getenv("RATE_LIMIT_MAX_VIOLATIONS", "10000"))

    def __init__(
        self,
        rate: float = DEFAULT_RATE,
        capacity: float = DEFAULT_CAPACITY,
        block_duration: float = DEFAULT_BLOCK_DURATION,
        cleanup_interval: float = 300.0,
    ):
        self.rate = rate
        self.capacity = capacity
        self.block_duration = block_duration
        self.buckets: dict[str, TokenBucket] = {}
        self.blocked_clients: dict[str, float] = {}
        self.violations: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._cleanup_interval = cleanup_interval
        self._bucket_access_times: dict[str, float] = {}

    async def start(self):
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError) as e:
                logger.warning("Rate limiter cleanup error: %s", e)

    async def _cleanup(self):
        now = time.time()
        async with self._lock:
            self.blocked_clients = {
                client: unblock_time
                for client, unblock_time in self.blocked_clients.items()
                if unblock_time > now
            }
            stale_threshold = now - 600
            stale_clients = [
                client
                for client, bucket in self.buckets.items()
                if bucket.last_update < stale_threshold
            ]
            for client in stale_clients:
                del self.buckets[client]

    def _get_client_id(self, request: web.Request) -> str:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
            if token:
                return f"apikey:{token[:16]}"
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"apikey:{api_key[:16]}"
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
            return f"ip:{client_ip}"
        peer_ip = request.remote or "unknown"
        return f"ip:{peer_ip}"

    async def is_allowed(self, request: web.Request) -> tuple[bool, dict]:
        """Check if request is allowed under rate limit"""

        client_id = self._get_client_id(request)
        now = time.time()
        async with self._lock:
            await self._enforce_limits()
            if client_id in self.blocked_clients:
                unblock_time = self.blocked_clients[client_id]
                if now < unblock_time:
                    wait_time = int(unblock_time - now)
                    headers = {"Retry-After": str(wait_time), "X-RateLimit-Blocked": "true"}
                    return False, headers
                else:
                    del self.blocked_clients[client_id]
                    self.violations[client_id] = 0
            if client_id not in self.buckets:
                self.buckets[client_id] = TokenBucket(
                    rate=self.rate,
                    capacity=self.capacity,
                    tokens=self.capacity,
                )
            bucket = self.buckets[client_id]
            self._bucket_access_times[client_id] = now
            if bucket.consume():
                remaining = int(bucket.tokens)
                reset_time = int(now + (1.0 - bucket.tokens) / self.rate + 1)
                headers = {
                    "RateLimit-Limit": str(int(self.capacity)),
                    "RateLimit-Remaining": str(max(0, remaining)),
                    "RateLimit-Reset": str(reset_time),
                    "RateLimit-Policy": f"{int(self.rate)};w=1",
                }
                return True, headers
            self.violations[client_id] += 1
            if self.violations[client_id] >= 5:
                self.blocked_clients[client_id] = now + self.block_duration
                logger.warning(
                    f"Rate limit abuse detected, blocking {client_id[:20]}... for {self.block_duration}s"
                )
            wait_time = bucket.get_wait_time()
            headers = {
                "Retry-After": str(int(wait_time) + 1),
                "RateLimit-Limit": str(int(self.capacity)),
                "RateLimit-Remaining": "0",
                "X-RateLimit-Violations": str(self.violations[client_id]),
            }
            return False, headers

    async def _enforce_limits(self):
        if len(self.buckets) > self.MAX_BUCKETS:
            sorted_clients = sorted(
                self._bucket_access_times.keys(), key=lambda c: self._bucket_access_times.get(c, 0)
            )
            to_remove = len(self.buckets) - self.MAX_BUCKETS
            for client in sorted_clients[:to_remove]:
                del self.buckets[client]
                del self._bucket_access_times[client]
                if client in self.violations:
                    del self.violations[client]
            logger.warning("Rate limiter evicted %s old buckets to enforce memory limit", to_remove)
        if len(self.violations) > self.MAX_VIOLATIONS_TRACKED:
            sorted_violations = sorted(self.violations.items(), key=lambda x: x[1])
            to_remove = len(self.violations) - self.MAX_VIOLATIONS_TRACKED
            for client, _ in sorted_violations[:to_remove]:
                del self.violations[client]
        now = time.time()
        if len(self.blocked_clients) > self.MAX_BLOCKED_CLIENTS:
            active_blocked = {k: v for k, v in self.blocked_clients.items() if v > now}
            if len(active_blocked) > self.MAX_BLOCKED_CLIENTS:
                sorted_blocked = sorted(active_blocked.items(), key=lambda x: x[1])
                to_remove = len(active_blocked) - self.MAX_BLOCKED_CLIENTS
                for client, _ in sorted_blocked[:to_remove]:
                    del self.blocked_clients[client]
            else:
                self.blocked_clients = active_blocked


# Instantiate at module import so aiohttp.AppKey sees a caller frame named "<module>".
RateLimiterAppKey = web.AppKey("rate_limiter", RateLimiter)


@web.middleware
async def rate_limit_middleware(request: web.Request, handler) -> web.StreamResponse:
    if os.getenv("RATE_LIMIT_DISABLED", "").lower() in ("1", "true", "yes"):
        return await handler(request)
    app = request.app
    limiter: RateLimiter | None = app.get(RateLimiterAppKey)
    if limiter is None:
        return await handler(request)
    if request.path == "/admin/health" or request.path == "/v1/health":
        return await handler(request)
    allowed, headers = await limiter.is_allowed(request)
    if not allowed:
        response = web.json_response(
            {"error": "Rate limit exceeded", "retry_after": headers.get("Retry-After", "1")},
            status=429,
        )
        for header, value in headers.items():
            response.headers[header] = value
        return response
    response = await handler(request)
    for header, value in headers.items():
        response.headers[header] = value
    return response


def create_rate_limiter(requests_per_second: float = 10.0, burst_size: float = 20.0) -> RateLimiter:
    import os

    rps = float(os.getenv("RATE_LIMIT_RPS", requests_per_second))
    burst = float(os.getenv("RATE_LIMIT_BURST", burst_size))
    return RateLimiter(rate=rps, capacity=burst)
