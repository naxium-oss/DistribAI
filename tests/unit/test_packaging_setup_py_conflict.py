"""Regression guard: nothing at repo root may be named ``setup.py``.

Context: this repo declares fully-static packaging metadata in
``pyproject.toml``'s ``[project]`` table (name, version, dependencies,
console `[project.scripts]` entries). setuptools' PEP 517 backend
(``setuptools.build_meta``) always `exec()`s a repo-root ``setup.py`` — if
one exists — as a legacy distutils entry point during *every*
``pip install .`` / ``pip install -e .`` / ``pip wheel .`` invocation,
passing it distutils subcommands like ``egg_info`` regardless of whether
``pyproject.toml`` metadata is otherwise sufficient. The interactive
packaging wizard previously lived at repo-root ``setup.py`` and had its own
unrelated argparse CLI, so every real install crashed with
``error: unrecognized arguments: egg_info`` before it was moved to
``scripts/packaging/setup_wizard.py``. See CHANGELOG.md "Unreleased".
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_no_setup_py_at_repo_root():
    assert not (_REPO_ROOT / "setup.py").exists(), (
        "A file named setup.py at the repo root breaks `pip install -e .` for "
        "everyone (setuptools execs it as a legacy distutils entry point). "
        "Put packaging scripts under scripts/packaging/ instead."
    )


@pytest.mark.unit
def test_packaging_setup_wizard_exists_at_new_home():
    wizard = _REPO_ROOT / "scripts" / "packaging" / "setup_wizard.py"
    assert wizard.is_file()
    assert "def main" in wizard.read_text(encoding="utf-8")


@pytest.mark.unit
def test_pyproject_declares_console_scripts_without_setup_py():
    """The [project.scripts] entry points must come from pyproject.toml alone."""
    try:
        import tomllib
    except ImportError:  # Python 3.10 fallback path (repo requires 3.11+, but be defensive)
        import tomli as tomllib  # type: ignore[import-not-found, no-redef]

    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["distribai"] == "scripts.cli.distribai_cli:main"
    assert scripts["distribai-tui"] == "scripts.cli.tui:main"
