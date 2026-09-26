"""SQLite database layer for DistribAI."""

from services_python.db._base import RETRYABLE_TASK_STATUSES, DBManagerBase
from services_python.db._credits import CreditsMixin
from services_python.db._jobs import JobsMixin
from services_python.db._nodes import NodesMixin
from services_python.db._tasks import TasksMixin
from services_python.db._votes import VotesMixin


class DBManager(
    NodesMixin,
    JobsMixin,
    CreditsMixin,
    TasksMixin,
    VotesMixin,
    DBManagerBase,
):
    """SQLite database manager for DistribAI"""


__all__ = ["DBManager", "RETRYABLE_TASK_STATUSES"]
