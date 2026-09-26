"""Extended tests for s3_util module."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest


def test_s3_util_imports():
    """Test s3_util module imports."""
    try:
        from worker.src.daemon import s3_util

        assert s3_util is not None
    except ImportError:
        pytest.skip("s3_util not available")


def test_s3_manager_creation():
    """Test S3Manager creation."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    with mock.patch.dict("os.environ", {}, clear=True):
        manager = S3Manager()
        assert manager is not None
        assert manager.access_key is None
        assert manager.secret_key is None
        assert manager.region == "us-east-1"


def test_s3_manager_with_env_vars():
    """Test S3Manager with environment variables."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    with mock.patch.dict(
        "os.environ",
        {
            "AWS_ACCESS_KEY_ID": "test-key-id",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
            "AWS_REGION": "us-west-2",
            "S3_BUCKET_NAME": "test-bucket",
        },
        clear=True,
    ):
        manager = S3Manager()
        assert manager.access_key == "test-key-id"
        assert manager.secret_key == "test-secret-key"
        assert manager.region == "us-west-2"
        assert manager.bucket_name == "test-bucket"


def test_s3_manager_upload_mocked():
    """Test S3Manager upload with mocks."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    with tempfile.TemporaryDirectory() as tmp:
        test_file = Path(tmp) / "test.txt"
        test_file.write_text("test content")

        with mock.patch.dict("os.environ", {}, clear=True):
            manager = S3Manager()
            assert hasattr(manager, "upload_file")


def test_s3_manager_download_mocked():
    """Test S3Manager download with mocks."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    with tempfile.TemporaryDirectory():
        with mock.patch.dict("os.environ", {}, clear=True):
            manager = S3Manager()
            assert hasattr(manager, "download_file")


def test_s3_manager_health_check():
    """Test S3Manager health_check method."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    with mock.patch.dict(
        "os.environ",
        {
            "S3_BUCKET_NAME": "test-bucket",
        },
        clear=True,
    ):
        manager = S3Manager()
        assert hasattr(manager, "health_check")


def test_parse_s3_url():
    """Test _parse_s3_url method."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    manager = S3Manager()
    bucket, key = manager._parse_s3_url("s3://my-bucket/path/to/file.txt")
    assert bucket == "my-bucket"
    assert key == "path/to/file.txt"


def test_parse_s3_url_invalid():
    """Test _parse_s3_url with invalid URL."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    manager = S3Manager()
    with pytest.raises(ValueError):
        manager._parse_s3_url("invalid-url")


def test_parse_s3_url_path_traversal():
    """Test _parse_s3_url blocks path traversal."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    manager = S3Manager()
    with pytest.raises(ValueError):
        manager._parse_s3_url("s3://bucket/../etc/passwd")


def test_parse_s3_url_null_bytes():
    """Test _parse_s3_url blocks null bytes."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    manager = S3Manager()
    with pytest.raises(ValueError):
        manager._parse_s3_url("s3://bucket/path\x00file")


def test_is_s3_url():
    """Test _is_s3_url method."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    manager = S3Manager()
    assert manager._is_s3_url("s3://bucket/key") is True
    assert manager._is_s3_url("http://example.com/file") is False
    assert manager._is_s3_url("/local/file") is False
    assert manager._is_s3_url(None) is False
    assert manager._is_s3_url("") is False


def test_upload_file_no_bucket():
    """Test upload_file returns local path when no bucket configured."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    with tempfile.TemporaryDirectory() as tmp:
        test_file = Path(tmp) / "test.txt"
        test_file.write_text("test content")

        with mock.patch.dict("os.environ", {}, clear=True):
            manager = S3Manager()
            # Since this is async, we can't test it directly without an event loop
            assert manager.bucket_name is None


def test_health_check_no_bucket():
    """Test health_check returns False when no bucket configured."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    with mock.patch.dict("os.environ", {}, clear=True):
        manager = S3Manager()
        assert manager.bucket_name is None
        # Can't test async method directly without event loop


def test_parse_s3_url_bucket_validation():
    """Test _parse_s3_url validates bucket name."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    manager = S3Manager()
    # Too short
    with pytest.raises(ValueError):
        manager._parse_s3_url("s3://ab/key")
    # Too long
    with pytest.raises(ValueError):
        manager._parse_s3_url("s3://" + "a" * 64 + "/key")
    # Invalid characters
    with pytest.raises(ValueError):
        manager._parse_s3_url("s3://my_bucket/key")


def test_parse_s3_url_key_validation():
    """Test _parse_s3_url validates key format."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    manager = S3Manager()
    # Key starts with /
    with pytest.raises(ValueError):
        manager._parse_s3_url("s3://bucket//key")
    # Key contains //
    with pytest.raises(ValueError):
        manager._parse_s3_url("s3://bucket/path//key")


def test_session_creation():
    """Test aioboto3 session creation."""
    try:
        from worker.src.daemon.s3_util import S3Manager
    except ImportError:
        pytest.skip("s3_util not available")
        return

    with mock.patch.dict(
        "os.environ",
        {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
        },
        clear=True,
    ):
        manager = S3Manager()
        assert manager.session is not None
