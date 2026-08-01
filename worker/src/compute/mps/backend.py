"""
MPS Compute Backend for Apple Silicon (M1/M2/M3)

Optimized for PyTorch Metal Performance Shaders.
"""

from __future__ import annotations

import gc
import json
import logging
import platform
import subprocess
from typing import Any

import psutil
import torch
import torch.optim as optim

from .. import ComputeBackend

logger = logging.getLogger(__name__)


class MPSBackend(ComputeBackend):
    def __init__(self, device_id: int = 0):
        super().__init__(device_id)
        self.device: torch.device | None = None
        self.name = "MPSBackend"

    def is_available(self) -> bool:
        if platform.system() != "Darwin":
            return False
        if not hasattr(torch.backends, "mps"):
            return False
        return torch.backends.mps.is_available()

    def initialize(self) -> bool:
        if not self.is_available():
            logger.error("MPS is not available on this system")
            return False
        try:
            self.device = torch.device("mps")
            test_tensor = torch.zeros(1, device="mps")
            _ = test_tensor + 1
            del test_tensor
            self._initialized = True
            logger.info(f"MPS backend initialized on {self.get_device_info()['name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MPS: {e}")
            return False

    def get_device_info(self) -> dict[str, Any]:
        if not self._initialized:
            return {"error": "Backend not initialized"}
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            cpu_name = result.stdout.strip()
        except (OSError, subprocess.TimeoutExpired, ValueError):
            cpu_name = "Apple Silicon"
        chip_gen = "Unknown"
        unified_memory_gb = 0
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            data = json.loads(result.stdout)
            hardware = data.get("SPHardwareDataType", [{}])[0]
            chip_name = hardware.get("chip_type", "")
            memory_bytes = hardware.get("physical_memory", 0)
            unified_memory_gb = memory_bytes // (1024**3) if memory_bytes else 0
            if "M3" in chip_name:
                chip_gen = "M3 (3rd Gen)"
            elif "M2" in chip_name:
                chip_gen = "M2 (2nd Gen)"
            elif "M1" in chip_name:
                chip_gen = "M1 (1st Gen)"
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
            pass

        return {
            "name": cpu_name,
            "chip_generation": chip_gen,
            "unified_memory_gb": unified_memory_gb,
            "total_memory_mb": unified_memory_gb * 1024,
            "supports_fp16": True,
            "supports_bf16": True,
            "is_apple_silicon": True,
            "mps_built": torch.backends.mps.is_built(),
        }

    def get_memory_stats(self) -> dict[str, int]:
        if not self._initialized:
            return {"error": "Backend not initialized"}
        mem = psutil.virtual_memory()
        return {
            "total_mb": mem.total // (1024 * 1024),
            "available_mb": mem.available // (1024 * 1024),
            "used_mb": mem.used // (1024 * 1024),
            "unified": True,
        }

    def optimize_model(self, model: Any) -> Any:
        if not self._initialized:
            raise RuntimeError("Backend not initialized")
        model = model.to(self.device)
        logger.info("Model moved to MPS device")
        return model

    def create_optimizer(self, model_parameters: Any, lr: float, **kwargs) -> Any:
        optimizer_class = kwargs.pop("optimizer_class", optim.AdamW)
        defaults = {
            "lr": lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
        defaults.update(kwargs)
        return optimizer_class(model_parameters, **defaults)

    def synchronize(self) -> None:
        if self._initialized and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def cleanup(self) -> None:
        if self._initialized:
            gc.collect()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            self._initialized = False
            logger.info("MPS backend cleaned up")
