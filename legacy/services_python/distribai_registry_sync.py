"""DistribAI model registry synchronization.

The registry is owned by DistribAI and is defined in the native model adapter.
This service exposes a stable orchestration boundary without depending on an
external repository or submodule.
"""

from __future__ import annotations

from copy import deepcopy


class DistribAIRegistrySync:
    """Read and optionally publish the first-party DistribAI model registry."""

    def __init__(self, orchestrator_db=None):
        self.db = orchestrator_db
        self.last_sync_ts = 0.0

    def sync(self) -> dict[str, dict]:
        """Return a snapshot of the native model configurations."""
        from worker.src.compute.distribai_models import DistribAIModelWrapper

        configs = deepcopy(DistribAIModelWrapper.MODEL_CONFIGS)
        self.last_sync_ts = __import__("time").time()
        return configs
