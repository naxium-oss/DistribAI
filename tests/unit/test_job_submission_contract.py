"""Regression tests for the explicit DistribAI script execution contract."""

from __future__ import annotations

import io
import tarfile

import pytest

from services_python.job_submission import (
    JobDistributor,
    JobPriority,
    JobSubmission,
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
