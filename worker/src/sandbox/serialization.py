"""Allowlisted pickle IPC between sandbox parent and child processes."""

from __future__ import annotations

import io
import pickle
from typing import Any

_ALLOWED_BUILTIN_NAMES = frozenset(
    {"list", "dict", "set", "int", "float", "str", "bool", "type", "tuple", "bytes"}
)

_ALLOWED_MODULES = frozenset(
    {
        "torch",
        "torch.nn",
        "torch.optim",
        "numpy",
        "collections",
        "datetime",
        "builtins",
        "worker.src.compute.distribai_models",
    }
)

_ALLOWED_TORCH_PREFIXES = (
    "torch.nn",
    "torch.nn.modules",
    "torch.nn.functional",
    "torch.optim",
    "torch.tensor",
    "torch._tensor",
)


def _module_allowed(module: str) -> bool:
    if module in _ALLOWED_MODULES:
        return True
    return module.startswith("torch.") and any(
        module == prefix or module.startswith(prefix + ".") for prefix in _ALLOWED_TORCH_PREFIXES
    )


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        if module == "builtins" and name in _ALLOWED_BUILTIN_NAMES:
            return super().find_class(module, name)
        if not _module_allowed(module):
            raise pickle.UnpicklingError(f"Global '{module}.{name}' is forbidden")
        return super().find_class(module, name)


def safe_loads(data: bytes) -> Any:
    return SafeUnpickler(io.BytesIO(data)).load()


def safe_dumps(obj: Any) -> bytes:
    """Parent/child payload dumps; children must load via safe_loads()."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def trusted_dumps(obj: Any) -> bytes:
    """Dump trusted entrypoints in the parent; child still uses safe_loads()."""
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
