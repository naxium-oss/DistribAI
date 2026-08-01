"""
Job management API for DistribAI
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .client import DistribAIError, JobNotFoundError, ValidationError

if TYPE_CHECKING:
    from .client import DistribAIClient


class JobStatus(Enum):
    """
    Status of a training job in the DistribAI.

    Attributes:
        QUEUED: Job is waiting in the queue for execution
        RUNNING: Job is currently being processed by worker nodes
        COMPLETED: Job finished successfully
        FAILED: Job failed due to an error
        CANCELLED: Job was cancelled by the user
        TIMEOUT: Job exceeded its time limit

    Example:
        status = JobStatus.RUNNING
        print(f"Job is {status.value}")
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """
    Represents a training job in DistribAI.
    Attributes:
        id: Unique job ID
        model_name: Model being trained (for example, "distribai-small")
        status: Current job status
        progress_pct: Progress percentage (0-100)
        current_step: Current training step
        total_steps: Total training steps
        credits_earned: Credits earned by contributors
        votes: Number of votes for this job
        queue_position: Position in queue (if queued)
        created_at: Creation timestamp
        started_at: Start timestamp (if started)
        finished_at: Finish timestamp (if finished)
    """

    id: str
    model_name: str
    status: JobStatus
    progress_pct: float
    current_step: int
    total_steps: int
    credits_earned: float = 0.0
    votes: int = 0
    queue_position: int | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    _client: DistribAIClient | None = None

    @classmethod
    def from_dict(cls, data: dict, client: DistribAIClient | None = None) -> Job:
        return cls(
            id=data["job_id"],
            model_name=data.get("model_name", "unknown"),
            status=JobStatus(data.get("status", "queued")),
            progress_pct=data.get("progress_pct", 0.0),
            current_step=data.get("current_step", 0),
            total_steps=data.get("total_steps", 0),
            credits_earned=data.get("credits_earned", 0.0),
            votes=data.get("total_votes", 0),
            queue_position=data.get("queue_position"),
            created_at=data.get("created_at"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            _client=client,
        )

    async def refresh(self) -> Job:
        """
        Refresh job status from API.
        Returns:
            Updated Job instance
        """
        if not self._client:
            raise RuntimeError("Job not associated with client")
        updated = await self._client.jobs.get(self.id)
        self.status = updated.status
        self.progress_pct = updated.progress_pct
        self.current_step = updated.current_step
        return self

    async def wait_for_completion(
        self,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> Job:
        """
        Wait for job to complete.
        Args:
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait (None = forever)
        Returns:
            Completed Job instance
        Raises:
            TimeoutError: If timeout exceeded
        """
        if not self._client:
            raise RuntimeError("Job not associated with client")
        start_time = asyncio.get_event_loop().time()
        while self.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise TimeoutError(f"Job {self.id} did not complete within {timeout}s")
            await asyncio.sleep(poll_interval)
            await self.refresh()
        return self

    async def cancel(self) -> bool:
        """
        Cancel this job.
        Returns:
            True if cancelled successfully
        """
        if not self._client:
            raise RuntimeError("Job not associated with client")
        return await self._client.jobs.cancel(self.id)

    @property
    def is_complete(self) -> bool:
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def eta_seconds(self) -> int | None:
        if self.status != JobStatus.RUNNING or self.progress_pct == 0:
            return None
        elapsed_ratio = self.progress_pct / 100.0
        if elapsed_ratio > 0:
            total_estimate = (asyncio.get_event_loop().time() - 0) / elapsed_ratio
            remaining = total_estimate * (1 - elapsed_ratio)
            return int(remaining)
        return None


class JobsAPI:
    def __init__(self, client: DistribAIClient):
        self._client = client

    async def submit(
        self,
        model_name: str,
        dataset: str,
        steps: int = 1000,
        batch_size: int = 32,
        priority: int = 5,
        hparams: dict | None = None,
    ) -> Job:
        """
        Submit a new training job.
        Args:
            model_name: Model to train ("distribai-small", "distribai-medium", "distribai-large", "custom", etc.)
            dataset: Dataset reference (S3 URL or HuggingFace dataset)
            steps: Number of training steps
            batch_size: Training batch size
            priority: Job priority (0-10, higher = more important)
            hparams: Additional hyperparameters
        Returns:
            Submitted Job instance
        Example:
            >>> job = await client.jobs.submit(
            ...     model_name="distribai-small",
            ...     dataset="s3://datasets/mydata.jsonl",
            ...     steps=5000
            ... )
            >>> print(f"Job submitted: {job.id}")
        """
        if steps < 1:
            raise ValidationError("steps must be positive")
        if batch_size < 1:
            raise ValidationError("batch_size must be positive")
        data = {
            "model_name": model_name,
            "dataset_ref": dataset,
            "steps": steps,
            "batch_size": batch_size,
            "priority": priority,
            "hparams": hparams or {},
        }
        response = await self._client._request("POST", "/v1/jobs", json=data)
        job_id = response.get("job_id")
        if job_id:
            return await self.get(job_id)
        return Job.from_dict(response, self._client)

    async def get(self, job_id: str) -> Job:
        """
        Get job by ID.
        Args:
            job_id: Job ID
        Returns:
            Job instance
        Raises:
            JobNotFoundError: If job not found
        """
        try:
            response = await self._client._request("GET", f"/v1/jobs/{job_id}")
            return Job.from_dict(response, self._client)
        except Exception as e:
            if "not found" in str(e).lower():
                raise JobNotFoundError(job_id) from e
            raise

    async def list(
        self,
        status: JobStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Job]:
        """
        List jobs.
        Args:
            status: Filter by status
            limit: Maximum jobs to return
            offset: Pagination offset
        Returns:
            List of Job instances
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status.value
        response = await self._client._request("GET", "/admin/jobs", params=params)
        jobs = response.get("jobs", [])
        return [Job.from_dict(j, self._client) for j in jobs]

    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a job.
        Args:
            job_id: Job ID to cancel
        Returns:
            True if cancelled successfully
        """
        try:
            await self._client._request("DELETE", f"/admin/jobs/{job_id}")
            return True
        except Exception:
            return False

    async def stream_logs(self, job_id: str) -> AsyncIterator[str]:
        """
        Stream job logs via SSE.
        Args:
            job_id: Job ID
        Yields:
            Log lines as they arrive
        """
        url = f"{self._client.base_url}/v1/jobs/{job_id}/logs/stream"
        headers = {"Accept": "text/event-stream"}
        if not self._client.session:
            raise RuntimeError("Client not connected. Call connect() first.")
        async with self._client.session.get(url, headers=headers) as response:
            if response.status != 200:
                raise DistribAIError(f"Failed to stream logs: {response.status}")
            buffer = ""
            async for chunk in response.content.iter_chunked(1024):
                buffer += chunk.decode("utf-8")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    for line in event.split("\n"):
                        if line.startswith("data: "):
                            data = line[6:]
                            if data.strip():
                                yield data
