from __future__ import annotations

from worker.src.daemon.redis_manager import RedisManager, RedisPubSub


def test_redis_manager_import():
    assert RedisManager is not None
    assert RedisPubSub is not None


def test_redis_manager_creation():
    manager = RedisManager()
    assert manager._in_memory_fallback is True


def test_in_memory_operations():
    manager = RedisManager()
    assert manager.set("key1", "value1") is True
    assert manager.get("key1") == "value1"


def test_in_memory_hash_operations():
    manager = RedisManager()
    assert manager.hset("hash1", "field1", "value1") is True
    assert manager.hget("hash1", "field1") == "value1"
    assert manager.hgetall("hash1") == {"field1": "value1"}


def test_in_memory_set_operations():
    manager = RedisManager()
    assert manager.sadd("set1", "member1") >= 0
    assert manager.sadd("set1", "member2") >= 0
    members = manager.smembers("set1")
    assert "member1" in members
    assert "member2" in members
    assert manager.sismember("set1", "member1") is True
    assert manager.sismember("set1", "member3") is False


def test_in_memory_sorted_set_operations():
    manager = RedisManager()
    assert manager.zadd("zset1", 1.0, "member1") is True
    assert manager.zadd("zset1", 2.0, "member2") is True
    assert manager.zrange("zset1", 0, -1) == ["member1", "member2"]
    assert manager.zrangebyscore("zset1", 1.5, 3.0) == ["member2"]


def test_in_memory_pubsub():
    manager = RedisManager()
    pubsub = RedisPubSub(manager)
    pubsub.subscribe("channel1", lambda ch, msg: None)
    assert "channel1" in pubsub.subscriptions
    assert len(pubsub.subscriptions["channel1"]) == 1


def test_exists_operation():
    manager = RedisManager()
    assert manager.exists("nonexistent") is False
    manager.set("existing", "value")
    assert manager.exists("existing") is True


def test_delete_operation():
    manager = RedisManager()
    manager.set("todelete", "value")
    manager.delete("todelete")
    assert manager.exists("todelete") is False


def test_incr_decr_operations():
    manager = RedisManager()
    manager.set("counter", "10")
    assert manager.incr("counter") == 11
    assert manager.decr("counter") == 10


def test_transaction():
    manager = RedisManager()

    def op1(m: RedisManager):
        m.set("tx_key1", "value1")

    def op2(m: RedisManager):
        m.set("tx_key2", "value2")

    assert manager.transaction([op1, op2]) is True
    assert manager.get("tx_key1") == "value1"
    assert manager.get("tx_key2") == "value2"
