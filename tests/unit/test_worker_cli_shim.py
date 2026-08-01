"""Unit tests for the worker/src/cli/main.py legacy-flag delegating shim.

Regression coverage for the bug where this module hard-imported `tabulate`
(never declared in requirements.txt/pyproject.toml) and crashed with
ModuleNotFoundError on every invocation of the registered `distribai`
console script on a clean install.
"""

from __future__ import annotations

import inspect
import sys
from unittest import mock

import pytest

import worker.src.cli.main as worker_cli_main
from worker.src.cli.main import _translate_submit, main


@pytest.mark.unit
def test_module_never_reimports_tabulate():
    """The whole point of this shim: no hard dependency on an undeclared package.

    (This module also already imported cleanly above, at test-collection
    time, which is the direct regression check — `tabulate` was never in
    requirements.txt/pyproject.toml, so a reintroduced `import tabulate`
    would fail collection for this entire file, not just this assertion.)
    """
    source_lines = inspect.getsource(worker_cli_main).splitlines()
    assert not any("import" in line and "tabulate" in line for line in source_lines)


@pytest.mark.unit
def test_translate_submit_maps_required_fields():
    args = mock.Mock(
        model="distribai-small",
        job_type="fine_tune",
        org="community",
        priority=0,
        priority_tier="P1",
        submitter_id="cli-user",
        steps=100,
        batch_size=32,
        deadline_seconds=600,
        steps_per_task=25,
        learning_rate=0.001,
        description="",
        weight_url="",
        batch_url="",
    )
    argv = _translate_submit(args)
    assert argv[:3] == ["job", "create", "distribai-small"]
    assert "100" in argv
    assert "--batch-size" in argv and "32" in argv
    assert "--org" in argv and "community" in argv
    # Optional/blank fields should not be forwarded.
    assert "--description" not in argv
    assert "--weight-url" not in argv
    assert "--batch-url" not in argv


@pytest.mark.unit
def test_translate_submit_forwards_optional_fields_when_set():
    args = mock.Mock(
        model="m",
        job_type="train",
        org="acme",
        priority=1,
        priority_tier="P0",
        submitter_id="alice",
        steps=10,
        batch_size=8,
        deadline_seconds=300,
        steps_per_task=5,
        learning_rate=0.01,
        description="desc",
        weight_url="s3://w",
        batch_url="s3://b",
    )
    argv = _translate_submit(args)
    assert "--description" in argv and "desc" in argv
    assert "--weight-url" in argv and "s3://w" in argv
    assert "--batch-url" in argv and "s3://b" in argv


@pytest.mark.unit
def test_main_nodes_delegates_to_consolidated_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["distribai", "nodes"])
    with mock.patch("worker.src.cli.main._consolidated_main") as delegated:
        main()
    delegated.assert_called_once_with(["nodes", "list"])


@pytest.mark.unit
def test_main_jobs_delegates_to_consolidated_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["distribai", "jobs"])
    with mock.patch("worker.src.cli.main._consolidated_main") as delegated:
        main()
    delegated.assert_called_once_with(["job", "list"])


@pytest.mark.unit
def test_main_submit_delegates_with_translated_argv(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["distribai", "submit", "--model", "m", "--steps", "5"])
    with mock.patch("worker.src.cli.main._consolidated_main") as delegated:
        main()
    delegated.assert_called_once()
    argv = delegated.call_args[0][0]
    assert argv[:3] == ["job", "create", "m"]
    assert "5" in argv


@pytest.mark.unit
def test_main_with_no_command_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["distribai"])
    main()
    assert "usage" in capsys.readouterr().out.lower()
