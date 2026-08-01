"""Pytest config for security tests.

Overrides the session-scoped event_loop fixture from tests/conftest.py so
that pytest-asyncio (in auto mode) creates a function-scoped loop. Without
this override, aiohttp TestClient instances built inside fixtures end up
attached to a different loop than the one the test body runs in, producing
"Timeout context manager should be used inside a task" errors.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def event_loop():
    """Function-scoped event loop -- one fresh loop per test."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()
