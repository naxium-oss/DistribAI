"""Tests for network benchmark module."""

import json


def get_module():
    from worker.src.benchmark import bench_network

    return bench_network


def test_emit(capsys):
    """Test emit function outputs JSON."""
    test_data = {"type": "test", "value": 42}
    get_module().emit(test_data)
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip()) == test_data


def test_log_score_basic():
    """Test log_score with normal values."""
    m = get_module()
    score = m.log_score(10.0, 1.0, 100.0)
    assert 0.0 <= score <= 100.0
    assert score > 0


def test_log_score_floor():
    """Test log_score at floor value returns 0."""
    m = get_module()
    score = m.log_score(1.0, 1.0, 100.0)
    assert score == 0.0


def test_log_score_invalid():
    """Test log_score with invalid inputs returns 0."""
    m = get_module()
    assert m.log_score(0.0, 1.0, 100.0) == 0.0
    assert m.log_score(10.0, 0.0, 100.0) == 0.0
    assert m.log_score(10.0, 5.0, 5.0) == 0.0


def test_log_score_very_high():
    """Test log_score clamps at 100."""
    m = get_module()
    score = m.log_score(1e6, 1.0, 100.0)
    assert score == 100.0


def test_emit_with_various_types():
    """Test emit handles various data types."""
    m = get_module()

    # Test with different data types
    test_cases = [
        {"type": "progress", "value": 50},
        {"type": "result", "score": 99.5},
        {"type": "error", "message": "test error"},
    ]

    for data in test_cases:
        # Should not raise
        m.emit(data)


def test_module_constants():
    """Test module has expected constants."""
    m = get_module()

    # Check for expected attributes
    assert hasattr(m, "emit")
    assert hasattr(m, "log_score")


def test_env_vars():
    """Test environment variable defaults."""
    m = get_module()
    assert m._FLOOR >= 0
    assert m._CEIL > m._FLOOR
    assert m._DL_DUR > 0
    assert m._LB_DUR > 0


def test_latency_hosts():
    """Test latency hosts list is populated."""
    m = get_module()
    assert len(m._LATENCY_HOSTS) > 0
    for host, port in m._LATENCY_HOSTS:
        assert isinstance(host, str)
        assert isinstance(port, int)
        assert port > 0


def test_bench_loopback():
    """Loopback throughput probe returns a non-negative Mbps reading."""
    m = get_module()
    assert hasattr(m, "bench_loopback")
    mbps = m.bench_loopback(0.2)
    assert isinstance(mbps, float)
    assert mbps >= 0.0


def test_bench_download_short():
    """Test download benchmark with short duration."""
    m = get_module()
    result, url = m.bench_download(duration=1.0)
    assert isinstance(result, (int, float))
    assert isinstance(url, str)


def test_try_download_rejects_non_https():
    m = get_module()
    assert m._try_download("http://example.com/file", 1.0) is None
    assert m._try_download("not-a-url", 1.0) is None


def test_bench_download_custom_duration():
    """Test download benchmark with custom duration."""
    m = get_module()
    result, url = m.bench_download(duration=0.5)
    assert isinstance(result, (int, float))
    assert isinstance(url, str)


def test_download_urls_are_https():
    m = get_module()
    assert m._DOWNLOAD_URLS
    for url in m._DOWNLOAD_URLS:
        assert url.startswith("https://")


def test_bench_download_timeout_handling():
    """Test download benchmark timeout handling."""
    m = get_module()
    result, url = m.bench_download(duration=0.1)
    assert isinstance(result, (int, float))
    assert isinstance(url, str)
