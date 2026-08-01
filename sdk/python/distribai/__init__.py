"""
DistribAI Python SDK
A clean, intuitive Python API for interacting with the DistribAI
distributed compute network.
Example:
    >>> import distribai
    >>>
    >>>
    >>> client = distribai.Client(api_key="cg_live_your_api_key_here")
    >>>
    >>>
    >>> job = client.jobs.submit(
    ...     model_name="distribai-small",
    ...     dataset="s3://datasets/mydata.jsonl",
    ...     steps=1000
    ... )
    >>>
    >>>
    >>> job.wait_for_completion()
    >>>
    >>>
    >>> credits = client.credits.balance()
    >>> print(f"Available credits: {credits.confirmed}")
"""

__version__ = "0.1.0"
__author__ = "DistribAI"
from .client import Client
from .credits import CreditBalance
from .exceptions import (
    AuthenticationError,
    DistribAIError,
    InsufficientCreditsError,
    JobNotFoundError,
    RateLimitError,
)
from .jobs import Job, JobStatus
from .nodes import Node, NodeStatus
from .votes import Vote

__all__ = [
    "Client",
    "Job",
    "JobStatus",
    "CreditBalance",
    "Node",
    "NodeStatus",
    "Vote",
    "DistribAIError",
    "AuthenticationError",
    "RateLimitError",
    "JobNotFoundError",
    "InsufficientCreditsError",
]
