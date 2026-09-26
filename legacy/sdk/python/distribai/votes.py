"""
Voting API for DistribAI
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import DistribAIClient


class VoteType(Enum):
    """
    Types of votes in the DistribAI governance system.

    Attributes:
        JOB_PRIORITY: Vote to change job priority
        MODEL_APPROVAL: Vote to approve a model for training
        PARAMETER_CHANGE: Vote to change system parameters
        NODE_REMOVAL: Vote to remove a malicious node

    Example:
        vote_type = VoteType.JOB_PRIORITY
        print(f"Vote type: {vote_type.value}")
    """

    JOB_PRIORITY = "job_priority"
    MODEL_APPROVAL = "model_approval"
    PARAMETER_CHANGE = "parameter_change"
    NODE_REMOVAL = "node_removal"


@dataclass
class Vote:
    """
    Represents a vote on a job.
    Attributes:
        job_id: Job being voted for
        credits_spent: Credits spent on this vote
        voter_id: Node ID that cast the vote
        timestamp: Vote timestamp
        vote_id: Unique vote ID
    """

    job_id: str
    credits_spent: int
    voter_id: str
    timestamp: str
    vote_id: str


@dataclass
class JobQueueItem:
    """
    Job in the voting queue.
    Attributes:
        job_id: Job ID
        title: Job title
        submitter: Job submitter
        total_votes: Total credits voted
        vote_count: Number of voters
        queue_position: Current position
        priority: Priority tier (P0-P3)
    """

    job_id: str
    title: str
    submitter: str
    total_votes: int
    vote_count: int
    queue_position: int | None
    priority: str = "P1"


class VotesAPI:
    def __init__(self, client: DistribAIClient):
        self._client = client

    async def cast(
        self,
        job_id: str,
        credits: int,
    ) -> dict:
        """
        Cast a vote on a job.
        Args:
            job_id: Job to vote for
            credits: Number of credits to spend (1 credit = 1 vote)
        Returns:
            Vote confirmation with new queue position
        Example:
            >>> result = await client.votes.cast("job_abc123", credits=100)
            >>> print(f"Job moved to position #{result['new_position']}")
        """
        data = {
            "job_id": job_id,
            "credits": credits,
        }
        response = await self._client._request("POST", "/v1/votes", json=data)
        return response

    async def list(self, limit: int = 20, offset: int = 0) -> list[Vote]:
        """
        List votes cast by the current user.
        Args:
            limit: Maximum votes to return
            offset: Pagination offset
        Returns:
            List of Vote instances
        """
        response = await self._client._request(
            "GET", "/v1/votes", params={"limit": limit, "offset": offset}
        )
        votes = response.get("votes", [])
        return [
            Vote(
                job_id=v["job_id"],
                credits_spent=v["credits"],
                voter_id=v["voter_id"],
                timestamp=v["timestamp"],
                vote_id=v["vote_id"],
            )
            for v in votes
        ]

    async def queue(self) -> list[JobQueueItem]:
        """
        Get the current job queue with vote tallies.
        Returns:
            List of JobQueueItem sorted by priority and votes
        Example:
            >>> queue = await client.votes.queue()
            >>> for job in queue[:5]:
            ...     print(f"{job.queue_position}. {job.title} ({job.total_votes} credits)")
        """
        response = await self._client._request("GET", "/v1/queue")
        jobs = response.get("jobs", response.get("queue", []))
        return [
            JobQueueItem(
                job_id=j["job_id"],
                title=j.get("title", "Untitled"),
                submitter=j.get("submitter", "Unknown"),
                total_votes=j.get("total_votes", 0),
                vote_count=j.get("vote_count", 0),
                queue_position=j.get("queue_position"),
                priority=j.get("priority", "P1"),
            )
            for j in jobs
        ]

    async def get_vote_tally(self, job_id: str) -> dict:
        """
        Get detailed vote tally for a specific job.
        Args:
            job_id: Job ID
        Returns:
            Dict with vote count, total credits, voter breakdown
        """
        response = await self._client._request("GET", f"/v1/votes/tally/{job_id}")
        return {
            "job_id": response.get("job_id"),
            "total_credits": response.get("total_credits", 0),
            "vote_count": response.get("vote_count", 0),
            "queue_position": response.get("queue_position"),
            "your_vote": response.get("your_vote", 0),
        }
