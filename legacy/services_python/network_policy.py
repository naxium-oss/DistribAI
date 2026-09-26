"""Shared network bind / loopback helpers."""

from __future__ import annotations

LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})


def is_loopback_host(host: str) -> bool:
    """True when host is a local-only admin/gRPC bind address."""
    return host.strip().lower() in LOOPBACK_HOSTS
