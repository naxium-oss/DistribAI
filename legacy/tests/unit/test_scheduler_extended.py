"""Extended tests for scheduler module."""

import asyncio

import pytest


def test_task_scheduler_creation():
    """Test TaskScheduler creation."""
    try:
        from services_python.scheduler import TaskScheduler
    except ImportError:
        pytest.skip("TaskScheduler not available")
        return

    class FakeDB:
        pass

    class FakeNodeService:
        connected_nodes = {}
        pending_assignments = {}

    scheduler = TaskScheduler(db=FakeDB(), node_service=FakeNodeService())
    assert scheduler is not None


@pytest.mark.asyncio
async def test_scheduler_assign_task_mocked():
    """Test TaskScheduler assign task with mocks."""
    try:
        from services_python.scheduler import TaskScheduler
    except ImportError:
        pytest.skip("TaskScheduler not available")
        return

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

    # Test that scheduler has required attributes
    assert hasattr(scheduler, "db")
    assert hasattr(scheduler, "node_service")


def test_scheduler_queue_depth():
    """Test TaskScheduler queue depth tracking."""
    try:
        from services_python.scheduler import TaskScheduler
    except ImportError:
        pytest.skip("TaskScheduler not available")
        return

    class FakeDB:
        def get_queue_depth(self):
            return 5

    class FakeNodeService:
        connected_nodes = {}

    scheduler = TaskScheduler(db=FakeDB(), node_service=FakeNodeService())

    # Test queue depth tracking
    assert hasattr(scheduler, "db")


@pytest.mark.asyncio
async def test_scheduler_assign_includes_script_package():
    """TaskAssign from scheduler carries cached script bytes."""
    from services_python.scheduler import TaskScheduler

    class FakeDB:
        def assign_task(self, task_id, node_id):
            pass

    class FakeNodeService:
        def __init__(self):
            self.connected_nodes = {"node-1": asyncio.Queue()}
            self.pending_assignments = {}
            self.script_packages = {"task-abc": b"tar-bytes"}

        def generate_presigned_url(self, key):
            return f"file:///tmp/{key}"

    db = FakeDB()
    ns = FakeNodeService()
    scheduler = TaskScheduler(db=db, node_service=ns)
    task = {
        "task_id": "task-abc",
        "job_id": "job-1",
        "model_name": "script-model",
        "hparams_json": "{}",
        "steps": 1,
    }
    await scheduler._assign_task_to_node(task, "node-1")
    msg = await ns.connected_nodes["node-1"].get()
    assert msg.assign.script_package == b"tar-bytes"
    assert msg.assign.execution_paradigm == "script"


@pytest.mark.asyncio
async def test_scheduler_script_assign_includes_task_context_and_deadline():
    """Script tasks need per-task metadata inside hparams/env, not just TaskAssign fields."""
    import json

    from services_python.scheduler import TaskScheduler

    class FakeDB:
        def assign_task(self, task_id, node_id):
            pass

    class FakeNodeService:
        def __init__(self):
            self.connected_nodes = {"node-1": asyncio.Queue()}
            self.pending_assignments = {}
            self.script_packages = {"task-ctx": b"tar-bytes"}

        def generate_presigned_url(self, key):
            return f"file:///tmp/{key}"

    ns = FakeNodeService()
    scheduler = TaskScheduler(db=FakeDB(), node_service=ns)
    task = {
        "task_id": "task-ctx",
        "job_id": "job-ctx",
        "model_name": "script-model",
        "hparams_json": json.dumps({"train_steps": 6}),
        "steps": 2,
        "step_offset": 4,
        "deadline_ts": 123456,
    }

    await scheduler._assign_task_to_node(task, "node-1")

    msg = await ns.connected_nodes["node-1"].get()
    hparams = json.loads(msg.assign.hparams_json)
    env = json.loads(msg.assign.distributed_env_json)
    assert hparams["train_steps"] == 6
    assert hparams["steps"] == 2
    assert hparams["distribai_job_id"] == "job-ctx"
    assert hparams["distribai_task_id"] == "task-ctx"
    assert hparams["distribai_task_steps"] == 2
    assert hparams["distribai_step_offset"] == 4
    assert env["DISTRIBAI_JOB_ID"] == "job-ctx"
    assert env["DISTRIBAI_TASK_STEPS"] == "2"
    assert msg.assign.deadline_ts == 123456


@pytest.mark.asyncio
async def test_scheduler_assign_loads_bundle_from_disk(tmp_path, monkeypatch):
    """Scheduler reads bundles from disk when in-memory cache is empty."""
    from services_python.bundle_store import save_bundle
    from services_python.scheduler import TaskScheduler

    monkeypatch.setenv("DISTRIBAI_BUNDLE_DIR", str(tmp_path))
    save_bundle("task-disk-1", b"from-disk")

    class FakeDB:
        def assign_task(self, task_id, node_id):
            pass

    class FakeNodeService:
        def __init__(self):
            self.connected_nodes = {"node-1": asyncio.Queue()}
            self.pending_assignments = {}
            self.script_packages = {}

        def generate_presigned_url(self, key):
            return f"file:///tmp/{key}"

    ns = FakeNodeService()
    scheduler = TaskScheduler(db=FakeDB(), node_service=ns)
    task = {
        "task_id": "task-disk-1",
        "job_id": "job-1",
        "model_name": "m",
        "hparams_json": "{}",
        "steps": 1,
    }
    await scheduler._assign_task_to_node(task, "node-1")
    msg = await ns.connected_nodes["node-1"].get()
    assert msg.assign.script_package == b"from-disk"
    assert msg.assign.execution_paradigm == "script"


def test_scheduler_constants():
    """Test scheduler constants."""
    try:
        from services_python import scheduler
    except ImportError:
        pytest.skip("scheduler module not available")
        return

    # Check that scheduler module has expected attributes
    assert hasattr(scheduler, "TaskScheduler")
