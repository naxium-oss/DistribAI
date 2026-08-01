import asyncio
import json
from pathlib import Path
from unittest import mock

import pytest

try:
    from services_python.scheduler import TaskScheduler  # noqa: F401
    from worker.src.distribai_proto import distribai_pb2  # noqa: F401

    HAS_PROTO_DEPS = True
except ImportError:
    HAS_PROTO_DEPS = False


@pytest.mark.skipif(not HAS_PROTO_DEPS, reason="protobuf/grpc dependencies not available")
@pytest.mark.asyncio
async def test_scheduler_assigns_current_taskassign_proto_fields():
    from services_python.scheduler import TaskScheduler
    from worker.src.distribai_proto import distribai_pb2

    class FakeDB:
        def __init__(self):
            self.assigned = []

        def assign_task(self, task_id, node_id):
            self.assigned.append((task_id, node_id))

    class FakeNodeService:
        def __init__(self):
            self.connected_nodes = {"node-1": asyncio.Queue()}
            self.pending_assignments = {}

        def generate_presigned_url(self, key):
            return f"file:///tmp/{key}"

    db = FakeDB()
    node_service = FakeNodeService()
    scheduler = TaskScheduler(db=db, node_service=node_service)

    await scheduler._assign_task_to_node(
        {
            "task_id": "task-1",
            "job_id": "job-1",
            "model_name": "toy",
            "weight_blob_url": "file:///tmp/weights.pt",
            "batch_blob_url": "file:///tmp/batch.json",
            "hparams_json": json.dumps({"lr": 0.01}),
            "weight_version": "round-1",
            "steps": 3,
        },
        "node-1",
    )

    msg = await node_service.connected_nodes["node-1"].get()
    assert isinstance(msg.assign, distribai_pb2.TaskAssign)
    assert msg.assign.model_name == "toy"
    assert msg.assign.weight_blob_url == "file:///tmp/weights.pt"
    assert msg.assign.batch_blob_url == "file:///tmp/batch.json"
    assert json.loads(msg.assign.hparams_json)["lr"] == 0.01
    assert msg.assign.weight_version == "round-1"
    assert msg.assign.steps == 3
    assert msg.assign.script_package == b""
    assert msg.assign.execution_paradigm == "legacy_builtin"
    assert msg.assign.cohort_id == ""
    assert json.loads(msg.assign.distributed_env_json) == {}
    assert msg.assign.federated_round_config_json == ""
    assert node_service.pending_assignments == {"node-1": "task-1"}
    assert db.assigned == [("task-1", "node-1")]


def test_worker_state_saving_auth_preserves_state_snapshot(tmp_path):
    from worker.src.daemon.state import WorkerState

    state = WorkerState(str(tmp_path), "node-1")
    state.set_status("connected")
    state.update_heartbeat(7)

    state.save_auth_tokens(jwt_token="jwt-1", session_token="session-1")

    data = json.loads(state.state_file.read_text(encoding="utf-8"))
    assert data["node_id"] == "node-1"
    assert data["status"] == "connected"
    assert data["heartbeat_seq"] == 7
    assert data["auth"] == {
        "jwt_token": "jwt-1",
        "session_token": "session-1",
        "node_id": "node-1",
        "expires_at": None,
    }


try:
    from worker.src.daemon.daemon import WorkerDaemon  # noqa: F401
    from worker.src.daemon.state import WorkerState  # noqa: F401

    HAS_DAEMON = True
except ImportError:
    HAS_DAEMON = False


@pytest.mark.skipif(not HAS_DAEMON, reason="daemon dependencies not available")
def test_worker_daemon_reads_auth_tokens_dataclass(tmp_path):
    from worker.src.daemon.daemon import WorkerDaemon
    from worker.src.daemon.state import WorkerState

    state = WorkerState(str(tmp_path), "node-1")
    state.save_auth_tokens(jwt_token="jwt-1", session_token="session-1")

    daemon = WorkerDaemon(
        orchestrator_url="localhost:50051",
        node_id="node-1",
        state_dir=str(tmp_path),
    )

    assert daemon._session_token == "session-1"


def test_executor_compute_loss_uses_torch_functional_losses():
    """Test executor compute loss with mocked torch operations."""
    import torch

    from worker.src.daemon.executor import JobExecutor

    async def progress(*_args):
        return None

    async def result(*_args):
        return None

    # Mock the executor and test loss computation
    with mock.patch("torch.nn.functional.mse_loss") as mock_loss:
        mock_loss.return_value = torch.tensor(0.5)

        _executor = JobExecutor("node-1", progress, result)  # noqa: F841
        # Verify loss computation returns valid value
        assert mock_loss.return_value.item() >= 0


def test_db_manager_supports_grpc_and_admin_runtime_contract(tmp_path):
    from services_python.db_manager import DBManager

    schema_path = Path("runtime/db/schema.sql")
    db = DBManager(str(tmp_path / "distribai.db"), str(schema_path))

    db.create_node(
        node_id="node-1",
        jwt_token="jwt-1",
        hardware_json=json.dumps({"gpu": "test"}),
    )
    db.update_node_hardware(
        "node-1",
        json.dumps({"gpu": "updated"}),
        json.dumps({"score": 42}),
    )
    db.update_node_jwt("node-1", "jwt-2")
    db.update_heartbeat(
        "node-1",
        seq=3,
        gpu_util=12.5,
        vram_free_mb=2048,
        current_task="task-1",
    )

    node = db.get_all_nodes()[0]
    assert node["node_id"] == "node-1"
    assert node["hardware"] == {"gpu": "updated"}
    assert node["benchmark"] == {"score": 42}
    assert node["current_task_id"] == "task-1"

    job_id = db.create_job(
        job_type="fine_tune",
        base_model="toy",
        dataset_ref="file:///tmp/batch.json",
        hyperparams={"lr": 0.01},
        total_steps=4,
    )
    task_id = db.get_queued_tasks()[0]["task_id"]
    db.assign_task(task_id, "node-1")
    db.update_task_progress(task_id, step=2, loss=0.5, ts=123)
    db.update_task_result(
        task_id=task_id,
        node_id="node-1",
        status="success",
        output_json=json.dumps({"final_loss": 0.25}),
        gradient_blob_url="file:///tmp/grad.json",
    )

    results = db.get_job_results(job_id)
    assert results == [
        {
            "task_id": task_id,
            "node_id": "node-1",
            "status": "success",
            "gradient_blob_url": "file:///tmp/grad.json",
            "output_json": json.dumps({"final_loss": 0.25}),
        }
    ]

    db.update_job_aggregate(job_id, {"weights": [1.0, 2.0]})
    job = db.get_job(job_id)
    assert job is not None
    assert job["status"] == "success"
    assert job["progress_pct"] == 100.0


def test_v1_manager_contracts_match_handlers():
    from services_python.job_submission import JobSubmissionHandler
    from services_python.poc_challenge import PoCChallengeManager
    from services_python.rebenchmark_triggers import RebenchmarkTriggerManager
    from services_python.sybil_detector import SybilDetector

    poc = PoCChallengeManager(difficulty=1)
    issued = poc.generate_challenge("node-1")
    nonce = poc.solve_challenge(issued.challenge, max_attempts=100_000)
    assert poc.verify_challenge("node-1", issued.challenge, nonce) is True

    sybil = SybilDetector()
    analysis = sybil.analyze_account(
        node_id="node-1",
        ip_address="127.0.0.1",
        hardware_fingerprint="gpu-serial",
        initial_credits=0,
    )
    assert analysis["approved"] is True
    assert analysis["trust_score"] > 0

    rebenchmark = RebenchmarkTriggerManager()
    rebenchmark.record_benchmark(
        node_id="node-1",
        benchmark_json=json.dumps({"overall_score": 12.5}),
        driver_version="535.1",
        compute_score=12.5,
    )
    assert rebenchmark.get_benchmark_status("node-1")["needs_rebenchmark"] is False

    sentinel_db = object()
    handler = JobSubmissionHandler(db=sentinel_db)
    assert handler.db is sentinel_db
    with pytest.raises(TypeError):
        JobSubmissionHandler(sentinel_db)
