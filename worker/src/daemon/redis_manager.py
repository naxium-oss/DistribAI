"""
Redis Integration for State Management (Production Implementation)
Provides Redis-backed state management for the DistribAI system,
supporting distributed state, caching, pub/sub messaging, streams,
sorted sets, transactions, and connection pooling.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class RedisPubSub:
    def __init__(self, redis_manager: RedisManager):
        self.redis_manager = redis_manager
        self.subscriptions: dict[str, list[Callable]] = defaultdict(list)
        self._running = False
        self._thread: threading.Thread | None = None

    def subscribe(self, channel: str, callback: Callable[[str, Any], None]):
        self.subscriptions[channel].append(callback)

    def unsubscribe(self, channel: str, callback: Callable | None = None):
        if callback:
            if callback in self.subscriptions[channel]:
                self.subscriptions[channel].remove(callback)
        else:
            self.subscriptions[channel].clear()

    def publish(self, channel: str, message: Any) -> bool:
        return self.redis_manager.publish(channel, message)

    def start(self):
        if not self._running and self.redis_manager.is_connected():
            self._running = True
            self._thread = threading.Thread(target=self._listen, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None

    def _listen(self):
        try:
            pubsub = self.redis_manager._client.pubsub()
            for channel in self.subscriptions:
                pubsub.subscribe(channel)
            while self._running:
                message = pubsub.get_message(timeout=1)
                if message and message["type"] == "message":
                    channel = message["channel"].decode()
                    data = message["data"].decode()
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        payload = data
                    for callback in self.subscriptions.get(channel, []):
                        try:
                            callback(channel, payload)
                        except (TypeError, ValueError, RuntimeError) as e:
                            logger.warning("PubSub callback error: %s", e)
        except (ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning("PubSub listener error: %s", e)


class RedisManager:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        max_connections: int = 10,
        decode_responses: bool = True,
    ):
        """
        Initialize Redis manager.
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            max_connections: Maximum connection pool size
            decode_responses: Whether to decode responses to strings
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.decode_responses = decode_responses
        self._client = None
        self._pool = None
        self._connected = False
        self._pubsub: RedisPubSub | None = None
        self._in_memory_store: dict[str, Any] = {}
        self._in_memory_sets: dict[str, set] = {}
        self._in_memory_sorted: dict[str, list[tuple[float, str]]] = {}
        self._in_memory_streams: dict[str, list[dict]] = {}

    @property
    def _in_memory_fallback(self) -> bool:
        return not self._connected

    @staticmethod
    def _coerce_int_counter(raw: Any) -> int:
        if raw is None:
            return 0
        if isinstance(raw, bool):
            return int(raw)
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
        if isinstance(raw, str):
            try:
                return int(raw.strip(), 10)
            except ValueError:
                return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    def connect(self) -> bool:
        try:
            import redis
            from redis.connection import ConnectionPool

            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=self.decode_responses,
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30,
            )
            self._client = redis.Redis(connection_pool=self._pool)
            self._client.ping()
            self._connected = True
            self._pubsub = RedisPubSub(self)
            return True
        except (ImportError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(
                "Redis connection failed: %s. Falling back to in-memory store. Data will not persist across restarts!",
                e,
            )
            self._in_memory_store: dict[str, Any] = {}
            self._in_memory_sets: dict[str, set] = {}
            self._in_memory_sorted: dict[str, list[tuple[float, str]]] = {}
            self._in_memory_streams: dict[str, list[dict]] = {}
            self._connected = False
            return False

    def is_connected(self) -> bool:
        return self._connected

    def get_pubsub(self) -> RedisPubSub | None:
        return self._pubsub

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        Set a key-value pair.
        Args:
            key: Key to set
            value: Value to store (will be JSON serialized)
            ttl: Time-to-live in seconds
        Returns:
            True if successful
        """
        if self._connected and self._client:
            try:
                serialized = json.dumps(value)
                if ttl:
                    return self._client.setex(key, ttl, serialized)
                else:
                    return self._client.set(key, serialized)
            except (TypeError, ValueError, ConnectionError):
                self._in_memory_store[key] = value
                return True
        else:
            self._in_memory_store[key] = value
            return True

    def get(self, key: str) -> Any | None:
        """
        Get a value by key.
        Args:
            key: Key to retrieve
        Returns:
            Deserialized value or None
        """
        if self._connected and self._client:
            try:
                value = self._client.get(key)
                if value:
                    return json.loads(value)
                return None
            except (TypeError, ValueError, ConnectionError):
                return self._in_memory_store.get(key)
        else:
            return self._in_memory_store.get(key)

    def delete(self, key: str) -> bool:
        """
        Delete a key.
        Args:
            key: Key to delete
        Returns:
            True if successful
        """
        if self._connected and self._client:
            try:
                return bool(self._client.delete(key))
            except ConnectionError:
                return self._in_memory_store.pop(key, None) is not None
        else:
            return self._in_memory_store.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        Args:
            key: Key to check
        Returns:
            True if key exists
        """
        if self._connected and self._client:
            try:
                return bool(self._client.exists(key))
            except ConnectionError:
                return key in self._in_memory_store
        else:
            return key in self._in_memory_store

    def hset(self, name: str, key: str, value: Any) -> bool:
        """
        Set a field in a hash.
        Args:
            name: Hash name
            key: Field key
            value: Field value
        Returns:
            True if successful
        """
        if self._connected and self._client:
            try:
                serialized = json.dumps(value)
                return bool(self._client.hset(name, key, serialized))
            except (TypeError, ValueError, ConnectionError):
                return False
        else:
            if name not in self._in_memory_store:
                self._in_memory_store[name] = {}
            if not isinstance(self._in_memory_store[name], dict):
                self._in_memory_store[name] = {}
            self._in_memory_store[name][key] = value
            return True

    def hget(self, name: str, key: str) -> Any | None:
        """
        Get a field from a hash.
        Args:
            name: Hash name
            key: Field key
        Returns:
            Field value or None
        """
        if self._connected and self._client:
            try:
                value = self._client.hget(name, key)
                if value:
                    return json.loads(value)
                return None
            except (TypeError, ValueError, ConnectionError):
                if isinstance(self._in_memory_store.get(name), dict):
                    return self._in_memory_store[name].get(key)
                return None
        else:
            if isinstance(self._in_memory_store.get(name), dict):
                return self._in_memory_store[name].get(key)
            return None

    def hgetall(self, name: str) -> dict[str, Any]:
        """
        Get all fields from a hash.
        Args:
            name: Hash name
        Returns:
            Dictionary of field-value pairs
        """
        if self._connected and self._client:
            try:
                data = self._client.hgetall(name)
                return {k: json.loads(v) for k, v in data.items()}
            except (TypeError, ValueError, ConnectionError):
                if isinstance(self._in_memory_store.get(name), dict):
                    return self._in_memory_store[name]
                return {}
        else:
            if isinstance(self._in_memory_store.get(name), dict):
                return self._in_memory_store[name]
            return {}

    def hdel(self, name: str, key: str) -> bool:
        """
        Delete a field from a hash.
        Args:
            name: Hash name
            key: Field key
        Returns:
            True if successful
        """
        if self._connected and self._client:
            try:
                return bool(self._client.hdel(name, key))
            except ConnectionError:
                if isinstance(self._in_memory_store.get(name), dict):
                    return self._in_memory_store[name].pop(key, None) is not None
                return False
        else:
            if isinstance(self._in_memory_store.get(name), dict):
                return self._in_memory_store[name].pop(key, None) is not None
            return False

    def sadd(self, name: str, *values: Any) -> int:
        """
        Add members to a set.
        Args:
            name: Set name
            values: Values to add
        Returns:
            Number of members added
        """
        if self._connected and self._client:
            try:
                serialized = [json.dumps(v) for v in values]
                return self._client.sadd(name, *serialized)
            except (TypeError, ValueError, ConnectionError):
                if name not in self._in_memory_sets:
                    self._in_memory_sets[name] = set()
                count = 0
                for v in values:
                    if v not in self._in_memory_sets[name]:
                        self._in_memory_sets[name].add(v)
                        count += 1
                return count
        else:
            if name not in self._in_memory_sets:
                self._in_memory_sets[name] = set()
            count = 0
            for v in values:
                if v not in self._in_memory_sets[name]:
                    self._in_memory_sets[name].add(v)
                    count += 1
            return count

    def smembers(self, name: str) -> set:
        """
        Get all members of a set.
        Args:
            name: Set name
        Returns:
            Set of members
        """
        if self._connected and self._client:
            try:
                members = self._client.smembers(name)
                return {json.loads(m) for m in members}
            except (TypeError, ValueError, ConnectionError):
                return self._in_memory_sets.get(name, set())
        else:
            return self._in_memory_sets.get(name, set())

    def srem(self, name: str, *values: Any) -> int:
        if self._connected and self._client:
            try:
                serialized = [json.dumps(v) for v in values]
                return self._client.srem(name, *serialized)
            except (TypeError, ValueError, ConnectionError):
                if name in self._in_memory_sets:
                    count = 0
                    for v in values:
                        if v in self._in_memory_sets[name]:
                            self._in_memory_sets[name].remove(v)
                            count += 1
                    return count
                return 0
        else:
            if name in self._in_memory_sets:
                count = 0
                for v in values:
                    if v in self._in_memory_sets[name]:
                        self._in_memory_sets[name].remove(v)
                        count += 1
                return count
            return 0

    def sismember(self, name: str, value: Any) -> bool:
        if self._connected and self._client:
            try:
                serialized = json.dumps(value)
                return bool(self._client.sismember(name, serialized))
            except (TypeError, ValueError, ConnectionError):
                return value in self._in_memory_sets.get(name, set())
        else:
            return value in self._in_memory_sets.get(name, set())

    def zadd(self, name: str, score: float, member: Any) -> bool:
        if self._connected and self._client:
            try:
                serialized = json.dumps(member)
                return bool(self._client.zadd(name, {serialized: score}))
            except (TypeError, ValueError, ConnectionError):
                if name not in self._in_memory_sorted:
                    self._in_memory_sorted[name] = []
                self._in_memory_sorted[name] = [
                    (s, m) for s, m in self._in_memory_sorted[name] if m != member
                ]
                self._in_memory_sorted[name].append((score, member))
                self._in_memory_sorted[name].sort(key=lambda x: x[0])
                return True
        else:
            if name not in self._in_memory_sorted:
                self._in_memory_sorted[name] = []
            self._in_memory_sorted[name] = [
                (s, m) for s, m in self._in_memory_sorted[name] if m != member
            ]
            self._in_memory_sorted[name].append((score, member))
            self._in_memory_sorted[name].sort(key=lambda x: x[0])
            return True

    def zrange(self, name: str, start: int = 0, end: int = -1) -> list[Any]:
        def _slice_in_memory(sorted_list: list[tuple[float, Any]]) -> list[Any]:
            if not sorted_list:
                return []
            n = len(sorted_list)
            if end == -1:
                stop = n
            else:
                stop = min(end + 1, n)
            start_i = max(0, min(start, n))
            stop_i = max(start_i, stop)
            return [m for _, m in sorted_list[start_i:stop_i]]

        if self._connected and self._client:
            try:
                members = self._client.zrange(name, start, end)
                return [json.loads(m) for m in members]
            except (TypeError, ValueError, ConnectionError):
                return _slice_in_memory(self._in_memory_sorted.get(name, []))
        else:
            return _slice_in_memory(self._in_memory_sorted.get(name, []))

    def zrangebyscore(self, name: str, min_score: float, max_score: float) -> list[Any]:
        if self._connected and self._client:
            try:
                members = self._client.zrangebyscore(name, min_score, max_score)
                return [json.loads(m) for m in members]
            except (TypeError, ValueError, ConnectionError):
                sorted_list = self._in_memory_sorted.get(name, [])
                return [m for s, m in sorted_list if min_score <= s <= max_score]
        else:
            sorted_list = self._in_memory_sorted.get(name, [])
            return [m for s, m in sorted_list if min_score <= s <= max_score]

    def zrem(self, name: str, member: Any) -> bool:
        if self._connected and self._client:
            try:
                serialized = json.dumps(member)
                return bool(self._client.zrem(name, serialized))
            except (TypeError, ValueError, ConnectionError):
                if name in self._in_memory_sorted:
                    original_len = len(self._in_memory_sorted[name])
                    self._in_memory_sorted[name] = [
                        (s, m) for s, m in self._in_memory_sorted[name] if m != member
                    ]
                    return len(self._in_memory_sorted[name]) < original_len
                return False
        else:
            if name in self._in_memory_sorted:
                original_len = len(self._in_memory_sorted[name])
                self._in_memory_sorted[name] = [
                    (s, m) for s, m in self._in_memory_sorted[name] if m != member
                ]
                return len(self._in_memory_sorted[name]) < original_len
            return False

    def publish(self, channel: str, message: Any) -> bool:
        if self._connected and self._client:
            try:
                serialized = json.dumps(message)
                return bool(self._client.publish(channel, serialized))
            except (TypeError, ValueError, ConnectionError):
                return False
        return False

    def xadd(self, name: str, fields: dict[str, Any], id: str | None = None) -> str | None:
        if self._connected and self._client:
            try:
                serialized_fields = {k: json.dumps(v) for k, v in fields.items()}
                return self._client.xadd(name, serialized_fields, id=id)
            except (TypeError, ValueError, ConnectionError):
                if name not in self._in_memory_streams:
                    self._in_memory_streams[name] = []
                entry = {
                    "id": id or str(int(time.time() * 1000)),
                    "fields": fields,
                    "timestamp": time.time(),
                }
                self._in_memory_streams[name].append(entry)
                return entry["id"]
        else:
            if name not in self._in_memory_streams:
                self._in_memory_streams[name] = []
            entry = {
                "id": id or str(int(time.time() * 1000)),
                "fields": fields,
                "timestamp": time.time(),
            }
            self._in_memory_streams[name].append(entry)
            return entry["id"]

    def xrange(self, name: str, count: int = 10) -> list[dict]:
        if self._connected and self._client:
            try:
                entries = self._client.xrange(name, count=count)
                result = []
                for entry_id, fields in entries:
                    result.append(
                        {"id": entry_id, "fields": {k: json.loads(v) for k, v in fields.items()}}
                    )
                return result
            except (TypeError, ValueError, ConnectionError):
                return self._in_memory_streams.get(name, [])[-count:]
        else:
            return self._in_memory_streams.get(name, [])[-count:]

    def transaction(self, operations: list[Callable]) -> bool:
        if self._connected and self._client:
            try:
                with self._client.pipeline() as pipe:
                    for op in operations:
                        op(pipe)
                    pipe.execute()
                return True
            except (ConnectionError, RuntimeError):
                return False
        else:
            for op in operations:
                op(self)
            return True

    def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment a value.
        Args:
            key: Key to increment
            amount: Amount to increment by
        Returns:
            New value
        """
        if self._connected and self._client:
            try:
                return self._client.incrby(key, amount)
            except (TypeError, ConnectionError):
                current = self._coerce_int_counter(self._in_memory_store.get(key, 0))
                self._in_memory_store[key] = current + amount
                return self._in_memory_store[key]
        else:
            current = self._coerce_int_counter(self._in_memory_store.get(key, 0))
            self._in_memory_store[key] = current + amount
            return self._in_memory_store[key]

    def ttl(self, key: str) -> int:
        if self._connected and self._client:
            try:
                return self._client.ttl(key)
            except ConnectionError:
                return -1
        return -1

    def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        if self._connected and self._client:
            try:
                return self._client.hincrby(name, key, amount)
            except (TypeError, ConnectionError):
                if name not in self._in_memory_store:
                    self._in_memory_store[name] = {}
                if not isinstance(self._in_memory_store[name], dict):
                    self._in_memory_store[name] = {}
                current = self._in_memory_store[name].get(key, 0)
                self._in_memory_store[name][key] = current + amount
                return self._in_memory_store[name][key]
        else:
            if name not in self._in_memory_store:
                self._in_memory_store[name] = {}
            if not isinstance(self._in_memory_store[name], dict):
                self._in_memory_store[name] = {}
            current = self._in_memory_store[name].get(key, 0)
            self._in_memory_store[name][key] = current + amount
            return self._in_memory_store[name][key]

    def decr(self, key: str, amount: int = 1) -> int:
        return self.incr(key, -amount)

    def keys(self, pattern: str = "*") -> list[str]:
        """
        Get all keys matching pattern.
        Args:
            pattern: Key pattern
        Returns:
            List of matching keys
        """
        if self._connected and self._client:
            try:
                return self._client.keys(pattern)
            except ConnectionError:
                return list(self._in_memory_store.keys())
        else:
            return list(self._in_memory_store.keys())

    def flushdb(self) -> bool:
        """
        Flush the current database.
        Returns:
            True if successful
        """
        if self._connected and self._client:
            try:
                return bool(self._client.flushdb())
            except ConnectionError:
                self._in_memory_store.clear()
                self._in_memory_sets.clear()
                return True
        else:
            self._in_memory_store.clear()
            self._in_memory_sets.clear()
            return True

    def disconnect(self):
        if self._pubsub:
            self._pubsub.stop()
            self._pubsub = None
        if self._client:
            try:
                self._client.close()
            except (ConnectionError, RuntimeError) as e:
                logger.debug("Redis client close error (ignored): %s", e)
            self._client = None
        if self._pool:
            try:
                self._pool.disconnect()
            except (ConnectionError, RuntimeError) as e:
                logger.debug("Redis pool disconnect error (ignored): %s", e)
            self._pool = None
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
