"""Regression tests for the explicit DistribAI script execution contract."""

from __future__ import annotations

import io
import tarfile

import pytest

from services_python.job_submission import (
    JobDistributor,
    JobPriority,
    JobSubmission,
    JobSubmissionHandler,
    JobType,
    validate_script_file,
)


def _job(**kwargs) -> JobSubmission:
    values = {
        "job_id": "job-contract",
        "org_id": "org-contract",
        "job_type": JobType.CUSTOM,
        "priority": JobPriority.NORMAL,
        "name": "contract test",
        "description": "script contract test",
    }
    values.update(kwargs)
    return JobSubmission(**values)


def test_script_file_validation_rejects_unsafe_source(tmp_path):
    script = tmp_path / "run.py"
    script.write_text("import subprocess\n", encoding="utf-8")

    errors, hints = validate_script_file(str(script))

    assert any(error.startswith("disallowed_import") for error in errors)
    assert hints


@pytest.mark.asyncio
async def test_package_script_requires_explicit_source():
    distributor = JobDistributor(queue=object(), node_service=object())

    with pytest.raises(ValueError, match="explicit script_content"):
        await distributor._package_script(_job())


@pytest.mark.asyncio
async def test_package_script_validates_and_contains_inline_source():
    distributor = JobDistributor(queue=object(), node_service=object())
    package = await distributor._package_script(_job(script_content="print('real work')\n"))

    with tarfile.open(fileobj=io.BytesIO(package), mode="r:gz") as archive:
        assert archive.extractfile("run.py").read() == b"print('real work')\n"
        assert "config.json" in archive.getnames()


@pytest.mark.asyncio
async def test_package_script_rejects_unsafe_inline_source():
    distributor = JobDistributor(queue=object(), node_service=object())

    with pytest.raises(ValueError, match="submitted script failed validation"):
        await distributor._package_script(_job(script_content="eval('fake')\n"))


def test_job_submission_handler_defaults_to_open_multi_tenant_orgs(monkeypatch):
    """Without DISTRIBAI_ALLOWED_ORGS, any non-empty org_id may use the API.

    Previously `allowed_orgs` started empty and nothing ever populated it
    (no admin endpoint called add_allowed_org), so every org was permanently
    rejected with 403 regardless of who submitted the job.
    """
    monkeypatch.delenv("DISTRIBAI_ALLOWED_ORGS", raising=False)
    handler = JobSubmissionHandler()

    assert handler._org_is_allowed("acme-corp") is True
    assert handler._org_is_allowed("any-other-org") is True
    assert handler._org_is_allowed("") is False
    assert handler._org_is_allowed(None) is False


def test_job_submission_handler_restricts_orgs_via_env(monkeypatch):
    monkeypatch.setenv("DISTRIBAI_ALLOWED_ORGS", "acme-corp, other-org")
    handler = JobSubmissionHandler()

    assert handler._org_is_allowed("acme-corp") is True
    assert handler._org_is_allowed("other-org") is True
    assert handler._org_is_allowed("unlisted-org") is False


def test_job_submission_handler_add_allowed_org_switches_to_restricted_mode(monkeypatch):
    monkeypatch.delenv("DISTRIBAI_ALLOWED_ORGS", raising=False)
    handler = JobSubmissionHandler()

    handler.add_allowed_org("only-this-org")

    assert handler._org_is_allowed("only-this-org") is True
    assert handler._org_is_allowed("some-other-org") is False
