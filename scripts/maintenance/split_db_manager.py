"""One-off helper to split db_manager.py into mixin modules."""

from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_PATH = ROOT / "services_python" / "db_manager.py"
OUT_DIR = ROOT / "services_python" / "db"

GROUPS: dict[str, list[str]] = {
    "_base.py": [
        "__init__",
        "_create_conn",
        "_ensure_conn",
        "_connect",
        "_init_schema",
        "_apply_migrations",
        "_ensure_column",
        "_safe_json",
    ],
    "_nodes.py": [
        "register_node",
        "create_node",
        "update_node_hardware",
        "get_node_jwt",
        "update_node_jwt",
        "update_heartbeat",
        "set_node_contributing",
        "update_node_benchmark",
        "get_all_nodes",
        "list_trusted_submitters",
        "add_trusted_submitter",
        "remove_trusted_submitter",
    ],
    "_jobs.py": [
        "insert_job",
        "create_job",
        "create_tasks",
        "insert_job_with_tasks",
        "refresh_queue_positions",
        "get_queued_tasks",
        "get_next_available_task",
        "get_queue_depth",
        "get_public_queue",
        "update_job_aggregate",
        "_refresh_job_state",
        "_job_total_steps",
        "operator_retry_job",
        "get_all_jobs",
        "get_job",
        "_job_row_to_dict",
        "update_job_status",
        "cancel_job",
    ],
    "_credits.py": [
        "add_credits",
        "transfer_credits_between_nodes",
        "get_node_credits",
        "list_all_credits",
        "iter_credit_ledger_rows",
    ],
    "_tasks.py": [
        "assign_task",
        "record_task_progress",
        "complete_task",
        "update_task_progress",
        "update_task_result",
        "get_job_results",
        "requeue_stale_tasks",
    ],
    "_votes.py": [
        "record_vote",
        "get_votes",
    ],
}

IMPORTS: dict[str, str] = {
    "_base.py": '''"""Connection, schema, and shared DBManager helpers."""

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)
RETRYABLE_TASK_STATUSES = {"error", "failed", "timeout", "invalid_gradient"}


''',
    "_nodes.py": '''"""Node registration and heartbeat mixin."""

import re
import time
from typing import Any


''',
    "_jobs.py": '''"""Job queue and lifecycle mixin."""

import json
import secrets
import sqlite3
import time
from typing import Any


''',
    "_credits.py": '''"""Credit ledger mixin."""

import time
from typing import Any


''',
    "_tasks.py": '''"""Task assignment and completion mixin."""

import sqlite3
import time
from typing import Any

from services_python.db._base import RETRYABLE_TASK_STATUSES


''',
    "_votes.py": '''"""Vote transaction mixin."""

import secrets
import time
from typing import Any


''',
}

CLASS_NAMES: dict[str, str] = {
    "_base.py": "DBManagerBase",
    "_nodes.py": "NodesMixin",
    "_jobs.py": "JobsMixin",
    "_credits.py": "CreditsMixin",
    "_tasks.py": "TasksMixin",
    "_votes.py": "VotesMixin",
}


def method_sources(src: str, cls: ast.ClassDef) -> dict[str, str]:
    lines = src.splitlines(keepends=True)
    out: dict[str, str] = {}
    for i, node in enumerate(cls.body):
        if not isinstance(node, ast.FunctionDef):
            continue
        start = node.lineno - 1
        if i + 1 < len(cls.body):
            end = cls.body[i + 1].lineno - 1
        else:
            end = len(lines)
        out[node.name] = "".join(lines[start:end])
    return out


def indent_methods(chunks: list[str], class_indent: int = 4) -> str:
    prefix = " " * class_indent
    parts: list[str] = []
    for chunk in chunks:
        lines = chunk.splitlines(keepends=True)
        dedented = "".join(line[4:] if line.startswith("    ") else line for line in lines)
        parts.append("".join(prefix + line if line.strip() else line for line in dedented.splitlines(keepends=True)))
    return "".join(parts)


def main() -> None:
    src = SRC_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DBManager")
    methods = method_sources(src, cls)
    OUT_DIR.mkdir(exist_ok=True)
    for fname, method_names in GROUPS.items():
        chunks = [methods[name] for name in method_names]
        if fname == "_base.py":
            body = indent_methods(chunks)
            content = (
                IMPORTS[fname]
                + f"class {CLASS_NAMES[fname]}:\n"
                + '    """SQLite database manager base: connection, schema, migrations."""\n\n'
                + body
            )
        else:
            body = indent_methods(chunks)
            content = (
                IMPORTS[fname]
                + f"class {CLASS_NAMES[fname]}:\n"
                + "    \"\"\"Mixin for DBManager.\"\"\"\n\n"
                + body
            )
        (OUT_DIR / fname).write_text(content, encoding="utf-8")
        print(f"wrote {fname}")


if __name__ == "__main__":
    main()
