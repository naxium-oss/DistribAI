"""
ROCm Compute Backend for AMD GPUs

Optimized for PyTorch with ROCm/HIP support.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from .. import ComputeBackend

logger = logging.getLogger(__name__)


class ROCmBackend(ComputeBackend):
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.name = "ROCmBackend"
        self.device: torch.device | None = None
        self._initialized = False

    def is_available(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            gpu_name = torch.cuda.get_device_name(0).lower()
            is_amd = any(x in gpu_name for x in ["amd", "radeon", "mi", " instinct"])
            is_hip = os.path.exists("/opt/rocm") or "HIP" in os.environ.get(
                "HSA_OVERRIDE_GFX_VERSION", ""
            )
            return is_amd or is_hip
        except (RuntimeError, OSError):
            return False

    def initialize(self) -> bool:
        if not self.is_available():
            logger.error("ROCm is not available on this system")
            return False
        try:
            torch.cuda.set_device(self.device_id)
            self.device = torch.device(f"cuda:{self.device_id}")
            torch.backends.cuda.matmul.allow_tf32 = True
            if "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
                pass
            self._initialized = True
            logger.info(f"ROCm backend initialized on {self.get_device_info()['name']}")
            return True
        except (RuntimeError, OSError) as e:
            logger.error("Failed to initialize ROCm: %s", e)
            return False

    def get_device_info(self) -> dict[str, Any]:
        if not self._initialized:
            return {"error": "Backend not initialized"}
        props = torch.cuda.get_device_properties(self.device_id)
        gpu_name = torch.cuda.get_device_name(self.device_id)
        arch_map = {
            "vega": (9, 0),
            "navi": (10, 1),
            "navi2": (10, 3),
            "navi3": (11, 0),
            "cdna": (9, 0),
            "mi100": (9, 0),
            "mi200": (9, 0),
            "mi250": (9, 0),
            "mi300": (9, 4),
        }
        arch = "Unknown"
        for key, _val in arch_map.items():
            if key in gpu_name.lower():
                arch = key.upper()
                break
        return {
            "name": gpu_name,
            "id": self.device_id,
            "total_memory_mb": props.total_memory // (1024 * 1024),
            "multi_processor_count": props.multi_processor_count,
            "architecture": arch,
            "supports_fp16": True,
            "supports_bf16": True,
            "rocm_version": self._get_rocm_version(),
        }

    def _get_rocm_version(self) -> str:
        try:
            rocm_path = "/opt/rocm/.info/version"
            if os.path.exists(rocm_path):
                with open(rocm_path) as f:
                    return f.read().strip()
        except OSError:
            pass
        return "unknown"

    def get_memory_stats(self) -> dict[str, int]:
        if not self._initialized:
            return {"error": "Backend not initialized"}
        torch_stats = torch.cuda.memory_stats(self.device_id)
        return {
            "total_mb": torch.cuda.get_device_properties(self.device_id).total_memory
            // (1024 * 1024),
            "allocated_mb": torch_stats.get("allocated_bytes.all.current", 0) // (1024 * 1024),
            "reserved_mb": torch_stats.get("reserved_bytes.all.current", 0) // (1024 * 1024),
            "free_mb": 0,
        }

    def optimize_model(self, model: Any) -> Any:
        if not self._initialized:
            raise RuntimeError("Backend not initialized")
        model = model.to(self.device)
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, backend="inductor")
                logger.info("Applied torch.compile optimization")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
        return model

    def create_optimizer(self, model_parameters: Any, lr: float, **kwargs):
        import torch.optim as optim

        optimizer_class = kwargs.pop("optimizer_class", optim.AdamW)
        defaults = {
            "lr": lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.01,
        }
        defaults.update(kwargs)
        return optimizer_class(model_parameters, **defaults)

    def synchronize(self):
        if self._initialized:
            torch.cuda.synchronize(self.device_id)

    def cleanup(self):
        if self._initialized:
            torch.cuda.empty_cache()
            self._initialized = False
            logger.info("ROCm backend cleaned up")
