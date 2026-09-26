"""Stable org/node identity persisted in ``~/.distribai/desktop.json``.

Mirrors ``client/lib/identityStore.js`` field-for-field (``org_id``,
``node_id``, ``node_name``) so the CLI/TUI, the Node.js dashboards, and the
pywebview desktop apps all agree on one identity per machine.
"""

from __future__ import annotations

import getpass
import re
import secrets

CONFIG_DIRNAME = ".distribai"
CONFIG_FILENAME = "desktop.json"


def new_org_id() -> str:
    return f"org-{secrets.token_hex(8)}"


def normalize_node_id(value: str | None) -> str:
    text = re.sub(r"\s+", "-", str(value or "").strip())
    return text.lower()


def ensure_identity(config: dict, *, username: str | None = None) -> tuple[dict, bool]:
    """Fill in missing ``org_id``/``node_name``/``node_id``; return (config, changed)."""
    next_config = dict(config)
    changed = False

    if not next_config.get("org_id"):
        next_config["org_id"] = new_org_id()
        changed = True

    if not next_config.get("node_name"):
        try:
            next_config["node_name"] = username or getpass.getuser()
            changed = True
        except OSError:
            pass

    if not next_config.get("node_id"):
        base = next_config.get("node_name") or username or "node"
        node_id = normalize_node_id(base)
        next_config["node_id"] = node_id or f"node-{secrets.token_hex(4)}"
        changed = True

    return next_config, changed
