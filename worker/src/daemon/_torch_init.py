"""PyInstaller runtime hook: fully pre-warm torch to break the init-time circular import.

PyInstaller bundles torch with its dynamic loader in a way that occasionally
causes `partially initialized module 'torch' has no attribute 'autograd'`
because torch's own `__init__.py` triggers nested submodules before its
top-level attribute table is filled in. We pre-import every submodule that
torch.__init__ will try to load, in the right order, so the eager import
inside `__init__.py` becomes a no-op.
"""

from __future__ import annotations

import importlib
import sys

_TORCH_PREWARM_ORDER = (
    "torch._C",
    "torch._VF",
    "torch.autograd",
    "torch.utils",
    "torch.utils.data",
    "torch.utils.checkpoint",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.jit",
    "torch.amp",
    "torch.cuda",
    "torch.linalg",
    "torch.signal",
    "torch.special",
    "torch.fft",
    "torch.random",
    "torch.distributions",
    "torch.multiprocessing",
    "torch.hub",
    "torch.serialization",
    "torch.fx",
    "torch.profiler",
    "torch.onnx",
    "torch.overrides",
    "torch.tensor",
    "torch.storage",
    "torch.dtype",
    "torch.layout",
    "torch.device",
    "torch.size",
    "torch.ops",
    "torch.classes",
    "torch._ops",
    "torch._subclasses",
    "torch.nested",
    "torch.sparse",
    "torch.ao",
    "torch.export",
    "torch.func",
    "torch.future",
)


def _prewarm_torch() -> None:
    import os
    debug = os.getenv("DISTRIBAI_DEBUG_TORCH_PREWARM") == "1"
    if debug:
        print(f"[prewarm] sys.path has {len(sys.path)} entries; _MEIPASS={getattr(sys, '_MEIPASS', None)}", flush=True)
    for name in _TORCH_PREWARM_ORDER:
        try:
            mod = importlib.import_module(name)
            sys.modules.setdefault(name, mod)
            if debug:
                print(f"[prewarm] ok: {name}", flush=True)
        except Exception as e:
            if debug:
                print(f"[prewarm] FAIL: {name}: {type(e).__name__}: {e}", flush=True)
            continue
    if debug:
        print(f"[prewarm] done; 'torch' in sys.modules: {'torch' in sys.modules}", flush=True)


_prewarm_torch()
