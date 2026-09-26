"""Tests for lightweight submitted-script validation."""

from services_python.script_validation import validate_submitted_script


def test_empty_script_no_errors():
    errs, hints = validate_submitted_script(None)
    assert errs == []
    assert hints == []


def test_valid_simple_script():
    src = "print('hello')\nx = 1 + 2\n"
    errs, hints = validate_submitted_script(src)
    assert errs == []
    assert hints == []


def test_syntax_error_reported():
    errs, hints = validate_submitted_script("def bad(\n")
    assert any(e.startswith("syntax_error:") for e in errs)
    assert hints


def test_subprocess_import_blocked():
    errs, hints = validate_submitted_script("import subprocess\nsubprocess.run(['echo'])\n")
    assert any("subprocess" in e for e in errs)
    assert hints


def test_disallowed_call_exec():
    errs, hints = validate_submitted_script("exec('1+1')\n")
    assert any("exec" in e for e in errs)
