import json
from types import SimpleNamespace

import pytest

try:
    from services_python.grpc_service import GrpcServiceHandler  # noqa: F401

    HAS_GRPC = True
except ImportError:
    HAS_GRPC = False


@pytest.mark.skipif(not HAS_GRPC, reason="grpc dependencies not available")
@pytest.mark.asyncio
async def test_gradient_payloads_load_and_bft_aggregate_live_format(tmp_path, monkeypatch):
    from services_python.grpc_service import GrpcServiceHandler

    monkeypatch.setenv("GRADIENT_LOCAL_ROOT", str(tmp_path))

    handler = GrpcServiceHandler(
        SimpleNamespace(
            db=None,
            byzantine_detector=None,
        )
    )

    gradient_path = tmp_path / "grad.json"
    gradient_path.write_text(
        json.dumps({"layer.bias": [1.0, 2.0], "layer.weight": [[3.0, 4.0]]}),
        encoding="utf-8",
    )

    loaded = await handler._load_gradient_payload(str(gradient_path))
    assert loaded == {"layer.bias": [1.0, 2.0], "layer.weight": [[3.0, 4.0]]}

    byzantine, aggregate = await handler._detect_byzantine_gradients(
        {
            "node-a": {"layer.bias": [1.0, 2.0], "layer.weight": [[3.0, 4.0]]},
            "node-b": {"layer.bias": [1.1, 2.1], "layer.weight": [[3.1, 4.1]]},
            "node-c": {"layer.bias": [50.0, 60.0], "layer.weight": [[70.0, 80.0]]},
            "node-d": {"layer.bias": [0.9, 1.9], "layer.weight": [[2.9, 3.9]]},
        }
    )

    assert aggregate is not None
    assert aggregate["method"] == "robust_bft_aggregate"
    assert set(aggregate["parameters"]) == {"layer.bias", "layer.weight"}
    assert aggregate["source_nodes"]
    assert isinstance(byzantine, bool)


def test_credit_ledger_live_credit_alias_is_signed():
    from worker.src.daemon.credit_ledger import CreditLedger

    ledger = CreditLedger(signing_key=b"phase2-test-key", batch_size=1)
    index = ledger.credit("node-a", 10.0, "task-1", {"reason": "task_success"})

    assert index == 0
    assert ledger.verify_chain_integrity()
    assert ledger.signature


def test_executor_writes_json_safe_compressed_gradient_envelope():
    """Test executor gradient envelope with proper compressed format."""
    from worker.src.daemon.executor import JobExecutor

    async def _noop(*args, **kwargs):
        return None

    executor = JobExecutor("node-a", _noop, _noop)

    # Test with proper compressed gradient format (indices, values, method, shape)
    compressed_data = {
        "layer.bias": {
            "indices": [0, 2],
            "values": [1.0, 2.0],
            "method": "topk",
            "shape": (3,),
        },
        "layer.weight": {
            "indices": [0, 3],
            "values": [3.0, 4.0],
            "method": "topk",
            "shape": (2, 2),
        },
    }

    compressed = executor._jsonify_compressed_gradients(compressed_data)
    envelope = {
        "compression": "dgc",
        "compressed": compressed,
    }

    # Verify envelope can be serialized to JSON
    json_str = json.dumps(envelope)
    assert envelope["compression"] == "dgc"
    assert "layer.bias" in envelope["compressed"]
    assert "layer.weight" in envelope["compressed"]
    assert json_str is not None
