"""Unit tests for admin_api.jobs module."""

import base64
import io
import json
import tarfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from services_python.admin_api.jobs import JobsHandler
from services_python.db_manager import DBManager


class TestJobsHandler:
    """Test cases for JobsHandler."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database manager."""
        db = MagicMock(spec=DBManager)
        return db

    @pytest.fixture
    def mock_node_service(self):
        """Create a mock node service."""
        node_service = MagicMock()
        node_service._authenticate_request = MagicMock()
        return node_service

    @pytest.fixture
    def jobs_handler(self, mock_db, mock_node_service):
        """Create a JobsHandler instance."""
        return JobsHandler(mock_db, mock_node_service)

    def test_init(self, mock_db, mock_node_service):
        """Test JobsHandler initialization."""
        handler = JobsHandler(mock_db, mock_node_service)
        assert handler.db == mock_db
        assert handler.node_service == mock_node_service

    @pytest.mark.asyncio
    async def test_list_jobs_success(self, jobs_handler, mock_db):
        """Test successful job listing."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.query.get.side_effect = lambda key, default="": {
            "include_history": "false",
            "active_only": "false",
        }.get(key, default)

        # Mock database response
        expected_jobs = [
            {"job_id": "job-1", "status": "running", "type": "train", "loss_history": [0.1, 0.05]},
            {
                "job_id": "job-2",
                "status": "completed",
                "type": "finetune",
                "loss_history": [0.2, 0.1],
            },
        ]
        mock_db.get_all_jobs.return_value = expected_jobs
        mock_db.get_queue_depth.return_value = 5

        # Call the handler
        response = await jobs_handler.list(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert "jobs" in data
        assert data["queue_depth"] == 5
        # Loss history should be removed when include_history is false
        assert "loss_history" not in data["jobs"][0]

        # Verify database was called
        mock_db.get_all_jobs.assert_called_once()

    @pytest.mark.asyncio
    async def test_compare_jobs_success(self, jobs_handler, mock_db):
        mock_request = MagicMock()
        mock_request.query.get.side_effect = lambda key, default="": {
            "a": "job-a",
            "b": "job-b",
        }.get(key, default)
        mock_db.get_job.side_effect = lambda jid: {
            "job-a": {
                "job_id": "job-a",
                "status": "failed",
                "model_name": "m1",
                "latest_reason": "hash mismatch",
            },
            "job-b": {
                "job_id": "job-b",
                "status": "success",
                "model_name": "m2",
                "latest_reason": "",
            },
        }.get(jid)

        response = await jobs_handler.compare(mock_request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["a"]["job_id"] == "job-a"
        assert data["b"]["status"] == "success"
        assert data["a"]["failure_code"] == "hash_mismatch"

    @pytest.mark.asyncio
    async def test_retry_job_requeues_tasks(self, jobs_handler, mock_db):
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "job-1"
        mock_db.get_job.return_value = {"job_id": "job-1", "status": "failed"}
        mock_db.operator_retry_job.return_value = {
            "job_id": "job-1",
            "requeued": ["task-9"],
        }

        response = await jobs_handler.retry(mock_request)
        assert response.status == 200
        data = json.loads(response.text)
        assert data["ok"] is True
        assert data["requeued_tasks"] == ["task-9"]
        mock_db.operator_retry_job.assert_called_once_with("job-1")

    @pytest.mark.asyncio
    async def test_retry_job_no_terminal_tasks(self, jobs_handler, mock_db):
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "job-1"
        mock_db.get_job.return_value = {"job_id": "job-1", "status": "failed"}
        mock_db.operator_retry_job.return_value = {"job_id": "job-1", "requeued": []}

        response = await jobs_handler.retry(mock_request)
        assert response.status == 409
        data = json.loads(response.text)
        assert "no terminal tasks" in data["error"]

    @pytest.mark.asyncio
    async def test_artifacts_lists_script_bundle(self, jobs_handler, mock_db, tmp_path, monkeypatch):
        import services_python.admin_api.jobs as jobs_mod

        bundle_dir = tmp_path / "bundles"
        bundle_dir.mkdir()
        monkeypatch.setattr(jobs_mod, "bundle_root", lambda: bundle_dir)

        task_id = "task_art1"
        (bundle_dir / f"{task_id}.tar.gz").write_bytes(b"fake")

        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "job-1"
        mock_db.get_job.return_value = {
            "job_id": "job-1",
            "latest_task_id": task_id,
        }

        response = await jobs_handler.artifacts(mock_request)
        assert response.status == 200
        data = json.loads(response.text)
        kinds = {item["kind"] for item in data["artifacts"]}
        assert "script_bundle" in kinds

    @pytest.mark.asyncio
    async def test_list_jobs_with_history(self, jobs_handler, mock_db):
        """Test job listing with history included."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.query.get.side_effect = lambda key, default="": {
            "include_history": "true",
            "active_only": "false",
        }.get(key, default)

        # Mock database response
        expected_jobs = [
            {"job_id": "job-1", "status": "running", "type": "train", "loss_history": [0.1, 0.05]}
        ]
        mock_db.get_all_jobs.return_value = expected_jobs
        mock_db.get_queue_depth.return_value = 3

        # Call the handler
        response = await jobs_handler.list(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert "jobs" in data
        assert data["queue_depth"] == 3
        # Loss history should be included when include_history is true
        assert "loss_history" in data["jobs"][0]

    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, jobs_handler, mock_db):
        """Test job listing when no jobs exist."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.query.get.side_effect = lambda key, default="": {
            "include_history": "false",
            "active_only": "false",
        }.get(key, default)

        # Mock database response - no jobs
        mock_db.get_all_jobs.return_value = []
        mock_db.get_queue_depth.return_value = 0

        # Call the handler
        response = await jobs_handler.list(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert data["jobs"] == []
        assert data["queue_depth"] == 0

    @pytest.mark.asyncio
    async def test_list_jobs_database_error(self, jobs_handler, mock_db):
        """Test job listing with database error."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.query.get.return_value = "false"

        # Mock database error
        mock_db.get_all_jobs.side_effect = Exception("Database error")

        # Call the handler - should raise exception
        with pytest.raises(Exception, match="Database error"):
            await jobs_handler.list(mock_request)

    @pytest.mark.asyncio
    async def test_get_job_success(self, jobs_handler, mock_db):
        """Test successful job retrieval."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "job-1"

        # Mock database response
        expected_job = {"job_id": "job-1", "status": "running", "type": "train"}
        mock_db.get_job.return_value = expected_job

        # Call the handler
        response = await jobs_handler.get(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert data["job_id"] == "job-1"
        assert data["status"] == "running"

        # Verify database was called
        mock_db.get_job.assert_called_once_with("job-1")

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, jobs_handler, mock_db):
        """Test job retrieval when job not found."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "nonexistent"

        # Mock database response - job not found
        mock_db.get_job.return_value = None

        # Call the handler
        response = await jobs_handler.get(mock_request)

        # Verify response
        assert response.status == 404
        data = json.loads(response.text)
        assert data["error"] == "not found"

    @pytest.mark.asyncio
    async def test_get_job_missing_job_id(self, jobs_handler):
        """Test job retrieval with missing job_id parameter."""
        # Setup mock request without job_id
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = None

        # Call the handler
        response = await jobs_handler.get(mock_request)

        # Verify response
        assert response.status == 400
        data = json.loads(response.text)
        assert data["error"] == "missing job_id"

    @pytest.mark.asyncio
    async def test_create_job_success(self, jobs_handler, mock_db):
        """Test successful job creation."""
        # Setup mock request with body using correct field names
        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "job_type": "fine_tune",
                "base_model": "gpt2",
                "dataset_ref": "custom_data",
                "steps": 1000,
                "batch_size": 32,
            }
        )

        # Mock database response
        mock_db.create_job.return_value = "job-123"
        mock_db.get_job.return_value = {"latest_task_id": "task-456"}

        # Call the handler
        response = await jobs_handler.create(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert data["ok"] is True
        assert data["job_id"] == "job-123"
        assert data["task_id"] == "task-456"

        # Verify database was called
        mock_db.create_job.assert_called_once()
        mock_db.get_job.assert_called_once_with("job-123")

    @pytest.mark.asyncio
    async def test_create_script_job_persists_bundle_for_all_tasks(
        self, jobs_handler, mock_db, mock_node_service, monkeypatch
    ):
        """Multi-task script jobs need the submitted package on every task."""
        import services_python.admin_api.jobs as jobs_mod

        saved: list[tuple[str, bytes]] = []

        def fake_save_bundle(task_id, package):
            saved.append((task_id, package))

        monkeypatch.setattr(jobs_mod, "save_bundle", fake_save_bundle)
        mock_node_service.script_packages = {}

        package = self._script_package()
        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "job_type": "fine_tune",
                "base_model": "gpt2",
                "steps": 6,
                "steps_per_task": 2,
                "script_package_b64": base64.b64encode(package).decode("ascii"),
            }
        )
        mock_db.create_job.return_value = "job-script-1"
        mock_db.get_job.return_value = {
            "latest_task_id": "task-a",
            "tasks": [
                {"task_id": "task-a"},
                {"task_id": "task-b"},
                {"task_id": "task-c"},
            ],
        }

        response = await jobs_handler.create(mock_request)

        assert response.status == 200
        assert [task_id for task_id, _ in saved] == ["task-a", "task-b", "task-c"]
        assert set(mock_node_service.script_packages) == {"task-a", "task-b", "task-c"}
        assert mock_db.create_job.call_args.kwargs["steps_per_task"] == 2

    @staticmethod
    def _script_package() -> bytes:
        buf = io.BytesIO()
        body = b"print('ok')\n"
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="run.py")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
        return buf.getvalue()

    @pytest.mark.asyncio
    async def test_list_paginated(self, jobs_handler, mock_db):
        """Test list jobs with pagination."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.query = {"page": "1", "per_page": "10", "sort": "job_id"}

        # Mock database response - should return a list, not a dict
        mock_jobs = [
            {
                "job_id": "job1",
                "status": "running",
                "created_ts": "2023-01-01T00:00:00Z",
                "updated_ts": "2023-01-01T01:00:00Z",
            },
            {
                "job_id": "job2",
                "status": "completed",
                "created_ts": "2023-01-02T00:00:00Z",
                "updated_ts": "2023-01-02T02:00:00Z",
            },
        ]
        mock_db.get_all_jobs.return_value = mock_jobs

        # Call the handler
        response = await jobs_handler.list_paginated(mock_request)

        # Verify response
        assert response.status == 200
        import json

        data = json.loads(response.text)
        assert "data" in data
        assert "pagination" in data
        assert len(data["data"]) == 2

        # Verify database was called
        mock_db.get_all_jobs.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_job_invalid_json(self, jobs_handler):
        """Test job creation with invalid JSON."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))

        # Call the handler
        response = await jobs_handler.create(mock_request)

        # Verify response
        assert response.status == 400
        data = json.loads(response.text)
        assert data["error"] == "invalid JSON"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_hparams", ["not-an-object", ["bad"], 7])
    async def test_create_job_rejects_non_object_hparams(self, jobs_handler, mock_db, bad_hparams):
        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={"base_model": "gpt2", "steps": 2, "hparams": bad_hparams}
        )

        response = await jobs_handler.create(mock_request)

        assert response.status == 400
        assert json.loads(response.text)["error"] == "hparams must be an object"
        mock_db.create_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_job_missing_fields(self, jobs_handler, mock_db):
        """Test job creation with missing required fields."""
        # Setup mock request with incomplete body
        mock_request = MagicMock()
        mock_request.json = AsyncMock(return_value={"job_type": "invalid_type"})  # Invalid job type

        # Call the handler
        response = await jobs_handler.create(mock_request)

        # Verify response - should return validation error
        assert response.status == 400
        data = json.loads(response.text)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_create_job_database_error(self, jobs_handler, mock_db):
        """Test job creation with database error."""
        # Setup mock request with valid body
        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "job_type": "fine_tune",
                "base_model": "gpt2",
                "dataset_ref": "custom_data",
                "steps": 1000,
            }
        )

        # Mock database error
        mock_db.create_job.side_effect = Exception("Database error")

        # Call the handler - should raise exception
        with pytest.raises(Exception, match="Database error"):
            await jobs_handler.create(mock_request)

    @pytest.mark.asyncio
    async def test_cancel_job_success(self, jobs_handler, mock_db):
        """Test successful job cancellation."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "job-1"

        mock_db.cancel_job = MagicMock(return_value=True)

        # Call the handler
        response = await jobs_handler.cancel(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert data["ok"] is True
        assert data["cancelled"] is True

        # Verify database was called
        mock_db.cancel_job.assert_called_once_with("job-1")

    @pytest.mark.asyncio
    async def test_cancel_job_not_found(self, jobs_handler, mock_db):
        """Test job cancellation when job not found."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "nonexistent"

        mock_db.cancel_job = MagicMock(return_value=False)

        response = await jobs_handler.cancel(mock_request)

        assert response.status == 404
        data = json.loads(response.text)
        assert data["cancelled"] is False

        mock_db.cancel_job.assert_called_once_with("nonexistent")

    @pytest.mark.asyncio
    async def test_cancel_job_database_error(self, jobs_handler, mock_db):
        """Test job cancellation with database error."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.match_info.get.return_value = "job-1"

        # Mock database error
        mock_db.cancel_job.side_effect = Exception("Database error")

        # Call the handler - should raise exception
        with pytest.raises(Exception, match="Database error"):
            await jobs_handler.cancel(mock_request)

    @pytest.mark.asyncio
    async def test_list_paginated_success(self, jobs_handler, mock_db):
        """Test successful paginated job listing."""
        # Setup mock request
        mock_request = MagicMock()
        mock_request.query = {}

        # Mock database response
        expected_jobs = [
            {"job_id": "job-1", "status": "running", "type": "train"},
            {"job_id": "job-2", "status": "completed", "type": "finetune"},
        ]
        mock_db.get_all_jobs.return_value = expected_jobs

        # Call the handler
        response = await jobs_handler.list_paginated(mock_request)

        # Verify response
        assert response.status == 200
        data = json.loads(response.text)
        assert "data" in data  # Changed from "items" to "data"
        assert "pagination" in data

        # Verify database was called
        mock_db.get_all_jobs.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_job_persists_callback_url_in_hparams(self, jobs_handler, mock_db):
        """hparams.callback_url is accepted and forwarded to create_job (delivery = phase 2)."""
        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "job_type": "fine_tune",
                "base_model": "gpt2",
                "dataset_ref": "custom_data",
                "steps": 10,
                "hparams": {"callback_url": "https://example.com/hook"},
            }
        )
        mock_db.create_job.return_value = "job-cb-1"
        mock_db.get_job.return_value = {"latest_task_id": "task-cb-1"}

        response = await jobs_handler.create(mock_request)

        assert response.status == 200
        _kwargs = mock_db.create_job.call_args.kwargs
        assert _kwargs["hyperparams"]["callback_url"] == "https://example.com/hook"

    @pytest.mark.asyncio
    async def test_create_job_rejects_forbidden_tarball(self, jobs_handler, mock_db):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, content in (("run.py", b"x=1\n"), (".env", b"SECRET=1\n")):
                info = tarfile.TarInfo(name=name)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "job_type": "fine_tune",
                "base_model": "gpt2",
                "dataset_ref": "local",
                "script_package_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
        )
        response = await jobs_handler.create(mock_request)
        assert response.status == 400
        data = json.loads(response.text)
        assert data["failure_code"] == "preflight_rejected"
        mock_db.create_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_job_rejects_invalid_script_content(self, jobs_handler, mock_db):
        mock_request = MagicMock()
        mock_request.json = AsyncMock(
            return_value={
                "job_type": "fine_tune",
                "base_model": "gpt2",
                "dataset_ref": "local",
                "script_content": "eval('bad')",
            }
        )
        response = await jobs_handler.create(mock_request)
        assert response.status == 400
        data = json.loads(response.text)
        assert data["failure_code"] == "script_validation_rejected"
        mock_db.create_job.assert_not_called()
