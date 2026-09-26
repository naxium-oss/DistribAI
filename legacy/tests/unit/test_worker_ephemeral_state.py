"""Unit tests for ephemeral worker state."""

from __future__ import annotations

import tempfile
from pathlib import Path

from worker.src.daemon.state import WorkerState


def test_ephemeral_auth_stays_in_memory_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state = WorkerState(tmp, "node-ephemeral", ephemeral=True)
        state.save_auth_tokens(jwt_token="jwt-test-token")
        loaded = state.load_auth_tokens()
        assert loaded.jwt_token == "jwt-test-token"
        state_file = Path(tmp) / "node-ephemeral" / "state.json"
        if state_file.exists():
            assert "jwt-test-token" not in state_file.read_text(encoding="utf-8")
        state.clear_auth_tokens()
        assert state.load_auth_tokens().jwt_token is None
