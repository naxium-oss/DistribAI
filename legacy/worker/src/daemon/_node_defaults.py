"""PyInstaller runtime hook for the DistribAI Node EXE.

This module runs at process start, before the main GUI script imports
anything else. It bakes in the public-grid defaults so the distributed
binary auto-connects to the operator's orchestrator without any user
configuration:

  * ORCHESTRATOR_URL  -- gRPC target the daemon dials
  * GRPC_TLS_CA       -- pinned root CA for TLS verification
  * GRPC_USE_TLS      -- always true on a public host
  * DISTRIBAI_LOCK_SERVER    -- suppress any user-facing server changes
  * DISTRIBAI_NODE_AUTOSTART -- register a Windows auto-start entry on first run

The defaults are written by `scripts/packaging/inject_public_host.py`
at build time so the CI workflow can swap them out per release.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PUBLIC_GRID_HOST: str = "167.160.87.208"
PUBLIC_GRID_PORT: str = "50051"
BUNDLED_CA_REL_PATH: str = "static/tls/ca.crt"


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
    return Path(__file__).resolve().parent


def apply_node_defaults() -> None:
    bundle_root = _bundle_root()
    ca_path = bundle_root / BUNDLED_CA_REL_PATH
    if ca_path.is_file():
        os.environ.setdefault("GRPC_TLS_CA", str(ca_path))
    os.environ.setdefault("ORCHESTRATOR_URL", f"{PUBLIC_GRID_HOST}:{PUBLIC_GRID_PORT}")
    os.environ.setdefault("GRPC_USE_TLS", "true")
    os.environ.setdefault("DISTRIBAI_LOCK_SERVER", "1")
    os.environ.setdefault("DISTRIBAI_NODE_AUTOSTART", "1")


apply_node_defaults()
