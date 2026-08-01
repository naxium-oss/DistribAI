"""
ML Core for Orchestrator - Gradient Aggregation and Checkpoint Management

Implements real gradient aggregation, checkpoint management, and model state
tracking for distributed training. No mocks - production implementation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class OrchestratorMLState:
    """
    Manages ML state for the orchestrator including gradient aggregation,
    checkpointing, and model version tracking.
    """

    def __init__(self, checkpoint_path: str):
        """
        Initialize ML state manager.
        Args:
            checkpoint_path: Path to store model checkpoints
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.step_count = 0
        self.model_state: dict[str, Any] = {}
        self.gradient_buffer: dict[str, torch.Tensor] = {}
        self.last_aggregation_time = 0.0
        self.aggregation_count = 0
        self.min_aggregation_count = 3  # Minimum gradients before aggregation
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if self.checkpoint_path.exists():
            try:
                checkpoint = torch.load(self.checkpoint_path, weights_only=True)
                self.step_count = checkpoint.get("step_count", 0)
                self.model_state = checkpoint.get("model_state", {})
                self.aggregation_count = checkpoint.get("aggregation_count", 0)
                logger.info(
                    f"Loaded checkpoint from {self.checkpoint_path}, step={self.step_count}"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}, starting fresh")

    def _save_checkpoint(self) -> None:
        try:
            checkpoint = {
                "step_count": self.step_count,
                "model_state": self.model_state,
                "aggregation_count": self.aggregation_count,
                "timestamp": time.time(),
            }
            torch.save(checkpoint, self.checkpoint_path)
            logger.debug(f"Saved checkpoint to {self.checkpoint_path}, step={self.step_count}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def apply_gradients(self, gradients_dict: dict[str, Any]) -> bool:
        """
        Apply gradients from a worker node.
        Args:
            gradients_dict: Dictionary of parameter names to gradient values
        Returns:
            True if gradients were applied successfully
        """
        try:
            gradient_tensors = {}
            for name, grad_list in gradients_dict.items():
                if isinstance(grad_list, list):
                    gradient_tensors[name] = torch.tensor(grad_list)
                else:
                    gradient_tensors[name] = grad_list
            if not self.gradient_buffer:
                self.gradient_buffer = gradient_tensors
            else:
                for name, grad in gradient_tensors.items():
                    if name in self.gradient_buffer:
                        self.gradient_buffer[name] += grad
                    else:
                        self.gradient_buffer[name] = grad
            self.aggregation_count += 1
            if self.aggregation_count >= self.min_aggregation_count:
                self._aggregate_and_step()
            return True
        except Exception as e:
            logger.error(f"Failed to apply gradients: {e}")
            return False

    def _aggregate_and_step(self) -> None:
        if not self.gradient_buffer:
            self.aggregation_count = 0
            return
        num_updates = self.aggregation_count
        for name in self.gradient_buffer:
            self.gradient_buffer[name] /= num_updates
        for name, grad in self.gradient_buffer.items():
            if name in self.model_state:
                lr = 0.01
                self.model_state[name] = self.model_state[name] - lr * grad
            else:
                self.model_state[name] = -grad * 0.01
        self.gradient_buffer.clear()
        self.aggregation_count = 0
        self.step_count += 1
        self.last_aggregation_time = time.time()
        if self.step_count % 10 == 0:
            self._save_checkpoint()

    def get_model_state(self) -> dict[str, Any]:
        """
        Get current model state for distribution to workers.
        Returns:
            Dictionary of model parameters
        """
        return {
            name: tensor.tolist() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in self.model_state.items()
        }

    def set_model_state(self, state_dict: dict[str, Any]) -> None:
        """
        Set model state (e.g., from loaded checkpoint).
        Args:
            state_dict: Dictionary of model parameters
        """
        self.model_state = {
            name: torch.tensor(value) if isinstance(value, list) else value
            for name, value in state_dict.items()
        }
        logger.info(f"Set model state with {len(self.model_state)} parameters")

    def get_checkpoint_info(self) -> dict[str, Any]:
        """
        Get information about current checkpoint state.
        Returns:
            Dictionary with checkpoint metadata
        """
        return {
            "step_count": self.step_count,
            "aggregation_count": self.aggregation_count,
            "last_aggregation_time": self.last_aggregation_time,
            "checkpoint_path": str(self.checkpoint_path),
            "parameter_count": len(self.model_state),
        }

    def reset(self) -> None:
        self.step_count = 0
        self.model_state.clear()
        self.gradient_buffer.clear()
        self.aggregation_count = 0
        self.last_aggregation_time = 0.0
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        logger.info("ML state reset to initial conditions")

    def force_checkpoint(self) -> None:
        self._save_checkpoint()
