"""
MyTrainer Sync for DistribAI

Watches the MyTrainer repository for changes and synchronizes
new model architectures with the Orchestrator and Workers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class MyTrainerSync:
    """
    Synchronizes model architectures from MyTrainer to DistribAI.

    Monitors the MyTrainer repository for changes to model configurations
    and architecture definitions, then propagates them to the orchestrator
    and worker nodes.

    Attributes:
        mytrainer_path: Path to the MyTrainer repository
        db: Orchestrator database manager
        last_sync_ts: Timestamp of last synchronization
        configs_file: Path to the grid architectures configuration file

    Example:
        sync = MyTrainerSync(
            mytrainer_path="/path/to/mytrainer",
            orchestrator_db=db_manager
        )
        sync.sync()
    """

    def __init__(self, mytrainer_path: str, orchestrator_db=None):
        """
        Initialize the MyTrainer sync manager.

        Args:
            mytrainer_path: Path to the MyTrainer repository
            orchestrator_db: Optional orchestrator database manager

        Example:
            >>> sync = MyTrainerSync(
            ...     mytrainer_path="/external/mytrainer",
            ...     orchestrator_db=db_manager
            ... )
        """
        self.mytrainer_path = Path(mytrainer_path)
        self.db = orchestrator_db
        self.last_sync_ts = 0.0
        self.configs_file = self.mytrainer_path / "configs" / "grid_architectures.json"

    def check_for_updates(self) -> bool:
        """
        Check if MyTrainer has been updated since last sync.

        Returns:
            True if updates are detected, False otherwise

        Example:
            >>> if sync.check_for_updates():
            ...     print("Updates detected, syncing...")
        """
        if not self.mytrainer_path.exists():
            logger.warning("MyTrainer path %s not found", self.mytrainer_path)
            return False
        if self.configs_file.exists():
            mtime = self.configs_file.stat().st_mtime
            if mtime > self.last_sync_ts:
                return True
        model_py = self.mytrainer_path / "models" / "model.py"
        if model_py.exists():
            if model_py.stat().st_mtime > self.last_sync_ts:
                return True
        return False

    def sync(self) -> dict:
        """
        Synchronize model architectures from MyTrainer.

        Loads architecture configurations and registers them with the
        the local DistribAI model registry and optionally the orchestrator database.

        Returns:
            Dictionary of loaded model configurations

        Example:
            >>> configs = sync.sync()
            >>> print(f"Synced {len(configs)} architectures")
        """
        logger.info("Synchronizing MyTrainer architectures...")
        configs = {}
        if self.configs_file.exists():
            try:
                with open(self.configs_file) as f:
                    configs = json.load(f)
                logger.info("Loaded %s architectures from %s", len(configs), self.configs_file.name)
            except (json.JSONDecodeError, OSError) as e:
                logger.error("Failed to load grid_architectures.json: %s", e)
        if os.getenv("DISTRIBAI_REGISTER_MYTRAINER_LOCALLY") == "1":
            try:
                from worker.src.compute.distribai_models import DistribAIModelWrapper

                for name, cfg in configs.items():
                    DistribAIModelWrapper.register_model_config(name, cfg)
            except ImportError:
                logger.warning(
                    "DistribAIModelWrapper not found in local path, skipping local registration"
                )
        if self.db:
            logger.info("Updating Orchestrator model registry...")
        self.last_sync_ts = time.time()
        return configs

    def start_watching(self, interval: int = 10) -> None:
        """
        Start watching for MyTrainer updates.

        Continuously polls for changes and syncs when updates are detected.

        Args:
            interval: Polling interval in seconds

        Example:
            >>> sync.start_watching(interval=30)
        """
        logger.info("Started watching %s (polling every %ss)", self.mytrainer_path, interval)
        while True:
            if self.check_for_updates():
                self.sync()
            time.sleep(interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).resolve().parents[1]
    mytrainer = root / "external" / "mytrainer"
    sync = MyTrainerSync(str(mytrainer))
    sync.sync()
