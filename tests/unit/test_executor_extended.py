"""Extended tests for executor module."""

import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from worker.src.daemon.executor import JobExecutor

    HAS_EXECUTOR = True
except ImportError:
    HAS_EXECUTOR = False
    JobExecutor = None


@pytest.mark.skipif(not HAS_EXECUTOR, reason="executor not available")
def test_job_executor_creation():
    """Test JobExecutor creation."""

    async def dummy_progress(*args):
        pass

    async def dummy_result(*args):
        pass

    executor = JobExecutor(
        node_id="test-node",
        on_progress=dummy_progress,
        on_result=dummy_result,
    )
    assert executor.node_id == "test-node"


@pytest.mark.skipif(not HAS_EXECUTOR or not HAS_TORCH, reason="executor or torch not available")
def test_executor_jsonify_compressed():
    """Test _jsonify_compressed_gradients method."""

    async def dummy_progress(*args):
        pass

    async def dummy_result(*args):
        pass

    executor = JobExecutor(
        node_id="test-node",
        on_progress=dummy_progress,
        on_result=dummy_result,
    )

    # Test with properly formatted compressed data
    compressed_data = {
        "layer.bias": {
            "indices": [0, 2],
            "values": [1.0, 2.0],
            "method": "topk",
            "shape": (3,),
        },
    }

    result = executor._jsonify_compressed_gradients(compressed_data)
    assert "layer.bias" in result
    assert "indices" in result["layer.bias"]


@pytest.mark.skipif(not HAS_EXECUTOR or not HAS_TORCH, reason="executor or torch not available")
def test_executor_toy_model_creation():
    """Test ToyModel creation."""
    from worker.src.daemon.executor import ToyModel

    model = ToyModel()
    assert model is not None

    # Test forward pass
    x = torch.randn(2, 10)
    y = model(x)
    assert y.shape == (2, 10)


@pytest.mark.skipif(not HAS_EXECUTOR or not HAS_TORCH, reason="executor or torch not available")
def test_executor_distribai_default_uses_native_model():
    """The schema default DistribAI profile should use the native model."""

    async def dummy_progress(*args):
        pass

    async def dummy_result(*args):
        pass

    executor = JobExecutor(
        node_id="test-node",
        on_progress=dummy_progress,
        on_result=dummy_result,
    )
    model = executor._create_model("distribai-small")
    assert model.__class__.__name__ == "DistribAIModelWrapper"


@pytest.mark.skipif(not HAS_EXECUTOR or not HAS_TORCH, reason="executor or torch not available")
def test_executor_compute_loss():
    """Test _compute_loss method."""

    async def dummy_progress(*args):
        pass

    async def dummy_result(*args):
        pass

    executor = JobExecutor(
        node_id="test-node",
        on_progress=dummy_progress,
        on_result=dummy_result,
    )

    from worker.src.daemon.executor import ToyModel

    model = ToyModel()

    # Create sample batch
    x = torch.randn(2, 10)
    y = torch.randn(2, 10)
    batch = (x, y)

    loss = executor._compute_loss(model, batch)
    assert loss is not None
    assert loss.item() >= 0


@pytest.mark.skipif(not HAS_EXECUTOR or not HAS_TORCH, reason="executor or torch not available")
def test_executor_compute_loss_moves_batch_to_model_device():
    """Batches must follow the model device (CPU always; CUDA when present)."""

    async def dummy_progress(*args):
        pass

    async def dummy_result(*args):
        pass

    executor = JobExecutor(
        node_id="test-node",
        on_progress=dummy_progress,
        on_result=dummy_result,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = executor._create_model("distribai-small").to(device)
    inputs = torch.randint(0, 256, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 8), dtype=torch.long)

    loss = executor._compute_loss(model, (inputs, targets))

    assert loss.device.type == device
    assert loss.item() >= 0


@pytest.mark.skipif(not HAS_EXECUTOR or not HAS_TORCH, reason="executor or torch not available")
def test_executor_detects_wrapped_compact_slm_as_language_model():
    """torch.compile-style wrappers expose the original module via _orig_mod."""

    async def dummy_progress(*args):
        pass

    async def dummy_result(*args):
        pass

    class Wrapped(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

        def forward(self, inputs):
            return self._orig_mod(inputs)

    executor = JobExecutor(
        node_id="test-node",
        on_progress=dummy_progress,
        on_result=dummy_result,
    )
    wrapped = Wrapped(executor._create_model("distribai-small"))
    inputs = torch.randint(0, 256, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 256, (2, 8), dtype=torch.long)

    assert executor._is_language_model(wrapped)
    loss = executor._compute_loss(wrapped, (inputs, targets))
    assert loss.item() >= 0


@pytest.mark.skipif(not HAS_EXECUTOR, reason="executor not available")
def test_executor_has_required_attributes():
    """Test JobExecutor has required attributes."""

    async def dummy_progress(*args):
        pass

    async def dummy_result(*args):
        pass

    executor = JobExecutor(
        node_id="test-node",
        on_progress=dummy_progress,
        on_result=dummy_result,
    )

    # Check required attributes
    assert hasattr(executor, "node_id")
    assert hasattr(executor, "on_progress")
    assert hasattr(executor, "on_result")
    assert hasattr(executor, "gradient_compressor")


@pytest.mark.skipif(not HAS_EXECUTOR, reason="executor not available")
@pytest.mark.asyncio
async def test_executor_reports_model_creation_error(monkeypatch):
    """Model factory errors must report a task result instead of escaping the task."""
    results = []

    async def dummy_progress(*args):
        pass

    async def capture_result(*args):
        results.append(args)

    executor = JobExecutor(
        node_id="test-node",
        on_progress=dummy_progress,
        on_result=capture_result,
    )
    executor.backend = None

    def raise_model_error(model_name):
        raise RuntimeError("model missing")

    monkeypatch.setattr(executor, "_create_model", raise_model_error)

    await executor.execute(
        {
            "job_id": "job-model-error",
            "task_id": "task-model-error",
            "model_name": "missing-model",
            "steps": 1,
            "batch_size": 1,
            "deadline_ts": 4_102_444_800,
        }
    )

    assert len(results) == 1
    job_id, task_id, status, _wall_ms, output = results[0]
    assert job_id == "job-model-error"
    assert task_id == "task-model-error"
    assert status == "error"
    assert output["error"] == "model missing"
