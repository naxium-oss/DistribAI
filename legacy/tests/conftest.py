"""
Pytest Configuration for DistribAI Tests

Provides fixtures for testing DistribAI components.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from tests.fast_env import fast_mode_enabled, wait_seconds


def pytest_configure(config: pytest.Config) -> None:
    os.environ.setdefault("DISTRIBAI_FAST_TEST", "1")
    os.environ.setdefault("DISTRIBAI_ALLOW_INSECURE_REGISTER", "1")
    os.environ.setdefault("DISTRIBAI_DB_DIR", tempfile.mkdtemp(prefix="distribai-pytest-db-"))
    os.environ.setdefault(
        "DISTRIBAI_BUNDLE_DIR", tempfile.mkdtemp(prefix="distribai-pytest-bundles-")
    )
    # Avoid real/misconfigured S3 during pytest (invalid keys break bundle + gradient paths).
    for _s3_var in (
        "S3_BUCKET_NAME",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        os.environ.pop(_s3_var, None)
    if fast_mode_enabled():
        os.environ.setdefault("MEM_DURATION_S", "0.05")
        os.environ.setdefault("BENCH_MEM_MAX_CHUNK_BYTES", str(64 * 2**20))
        os.environ.setdefault("DISTRIBAI_HEALTH_CHECK_TIMEOUT", "1.5")
        os.environ.setdefault("WRITE_DURATION_S", "0.05")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.skipped:
        pytest.fail(f"Skipped tests are not allowed: {report.longrepr}")


@pytest.fixture(scope="session", autouse=True)
def _accelerate_blocking_waits():
    """Cap long time.sleep calls when DISTRIBAI_FAST_TEST=1 (default)."""
    if not fast_mode_enabled():
        yield
        return
    real_sleep = time.sleep

    def capped_sleep(seconds: float, *args, **kwargs) -> None:
        real_sleep(wait_seconds(float(seconds)))

    time.sleep = capped_sleep
    yield
    time.sleep = real_sleep


@pytest.fixture(scope="session", autouse=True)
def _accelerate_async_waits():
    if not fast_mode_enabled():
        yield
        return
    real_sleep = asyncio.sleep

    async def capped_sleep(delay: float, result=None):
        await real_sleep(wait_seconds(float(delay)), result)

    asyncio.sleep = capped_sleep
    yield
    asyncio.sleep = real_sleep


@pytest.fixture
def temp_dir():
    """
    Create a temporary directory for tests.

    Yields:
        Path to temporary directory

    Example:
        >>> def test_something(temp_dir):
        ...     assert Path(temp_dir).exists()
    """
    tmp = tempfile.mkdtemp()
    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def temp_state_dir(temp_dir):
    """
    Create a temporary state directory.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to state directory

    Example:
        >>> def test_state(temp_state_dir):
        ...     assert Path(temp_state_dir).exists()
    """
    state_dir = Path(temp_dir) / "state"
    state_dir.mkdir()
    return str(state_dir)


@pytest.fixture
def temp_checkpoint_dir(temp_dir):
    """
    Create a temporary checkpoint directory.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path to checkpoint directory

    Example:
        >>> def test_checkpoint(temp_checkpoint_dir):
        ...     assert Path(temp_checkpoint_dir).exists()
    """
    ckpt_dir = Path(temp_dir) / "checkpoints"
    ckpt_dir.mkdir()
    return str(ckpt_dir)


@pytest.fixture
def mock_node_id():
    """
    Provide a mock node ID for testing.

    Returns:
        Test node identifier

    Example:
        >>> def test_node(mock_node_id):
        ...     assert mock_node_id == "test-node-12345"
    """
    return "test-node-12345"


@pytest.fixture
def sample_gradients():
    """
    Provide sample gradient tensors for testing.

    Returns:
        Dictionary of parameter name to gradient tensor

    Example:
        >>> def test_grads(sample_gradients):
        ...     assert "layer1.weight" in sample_gradients
    """
    import torch

    return {
        "layer1.weight": torch.randn(10, 10),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
    }


@pytest.fixture
def simple_model():
    """
    Provide a simple neural network model for testing.

    Returns:
        Simple sequential model

    Example:
        >>> def test_model(simple_model):
        ...     output = simple_model(torch.randn(1, 10))
        ...     assert output.shape == (1, 10)
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )


@pytest.fixture(scope="session")
def event_loop():
    """
    Provide an asyncio event loop for async tests.

    Session-scoped fixture that creates and manages the event loop.

    Yields:
        Asyncio event loop

    Example:
        >>> async def test_async(event_loop):
        ...     result = await some_async_function()
    """
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_jwt_token():
    """
    Provide a mock JWT token for testing.

    Returns:
        Valid JWT token for test authentication

    Example:
        >>> def test_auth(mock_jwt_token):
        ...     assert mock_jwt_token is not None
    """
    import os
    import time

    import jwt
    from dotenv import load_dotenv

    load_dotenv()
    jwt_secret = os.getenv("JWT_SECRET", "test-secret-key")

    payload = {
        "node_id": "test-node-12345",
        "exp": int(time.time()) + 3600,  # 1 hour expiration
        "iat": int(time.time()),
        "type": "node",
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")
