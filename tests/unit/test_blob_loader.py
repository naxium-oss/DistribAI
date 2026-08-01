"""Tests for canonical blob JSON loader."""

from __future__ import annotations

import pytest

from services_python.blob_loader import load_json_blob, load_text_blob


@pytest.mark.asyncio
async def test_load_json_blob_blocks_disallowed_host(monkeypatch):
    monkeypatch.setenv("ALLOWED_BLOB_HOSTS", "127.0.0.1,localhost")
    result = await load_json_blob("https://evil.example/grad.json")
    assert result is None


@pytest.mark.asyncio
async def test_load_text_blob_blocks_disallowed_host(monkeypatch):
    monkeypatch.setenv("ALLOWED_BLOB_HOSTS", "127.0.0.1,localhost")
    result = await load_text_blob("https://evil.example/batch.txt")
    assert result is None
