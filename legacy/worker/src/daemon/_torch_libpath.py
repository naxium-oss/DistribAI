"""PyInstaller runtime hook: prepend the bundled torch lib dir to LD_LIBRARY_PATH
so that torch's ctypes-based _load_global_deps finds libtorch_global_deps.so
even though PyInstaller keeps it under _internal/torch/lib/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    meipass = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    torch_lib = meipass / "torch" / "lib"
    if torch_lib.is_dir():
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            f"{torch_lib}{os.pathsep}{existing}" if existing else str(torch_lib)
        )
        # Try the same on macOS for good measure
        existing_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["DYLD_LIBRARY_PATH"] = (
            f"{torch_lib}{os.pathsep}{existing_dyld}" if existing_dyld else str(torch_lib)
        )
