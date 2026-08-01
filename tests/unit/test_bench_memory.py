"""Tests for memory benchmark module."""

from unittest import mock


def get_module():
    from worker.src.benchmark import bench_memory

    return bench_memory


def test_emit_outputs_json(capsys):
    """Test emit function outputs JSON."""
    import json

    test_data = {"type": "test"}
    get_module().emit(test_data)
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip()) == test_data


def test_log_score_returns_float():
    """Test log_score returns a valid score."""
    m = get_module()
    score = m.log_score(10.0, m._FLOOR, m._CEIL)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_log_score_clamps_high():
    """Test log_score clamps at 100."""
    m = get_module()
    score = m.log_score(1000.0, m._FLOOR, m._CEIL)
    assert score == 100.0


def test_log_score_clamps_low():
    """Test log_score returns 0 for invalid input."""
    m = get_module()
    score = m.log_score(0.0, m._FLOOR, m._CEIL)
    assert score == 0.0


def test_constants_valid():
    """Test benchmark constants are valid."""
    m = get_module()
    assert m._DUR > 0
    assert m._FLOOR > 0
    assert m._CEIL > m._FLOOR


def test_bench_latency_cross_platform():
    """Test latency benchmark on all platforms."""
    m = get_module()
    result = m.bench_latency(n_pointers=1024, n_walks=100)
    assert isinstance(result, (int, float, type(None)))
    if result is not None:
        assert result > 0


def test_bench_latency_mocked():
    """Test latency benchmark with mocked time."""
    m = get_module()
    with mock.patch.object(m, "time") as mock_time:
        mock_time.perf_counter.side_effect = [0.0, 1.0]
        result = m.bench_latency()
        assert isinstance(result, (int, float, type(None)))


def test_bench_sequential_write_mocked():
    """Test sequential write benchmark."""
    m = get_module()
    with mock.patch.object(m, "time") as mock_time:
        mock_time.perf_counter.side_effect = [0.0, 0.05, 0.1, 0.15]
        result = m.bench_sequential_write(1024, duration=0.1)
        assert isinstance(result, (int, float, type(None)))


def test_bench_sequential_read_mocked():
    """Test sequential read benchmark."""
    m = get_module()
    with mock.patch.object(m, "time") as mock_time:
        mock_time.perf_counter.side_effect = [0.0, 0.05, 0.1, 0.15]
        result = m.bench_sequential_read(1024, duration=0.1)
        assert isinstance(result, (int, float, type(None)))


def test_bench_copy_mocked():
    """Test copy benchmark."""
    m = get_module()
    with mock.patch.object(m, "time") as mock_time:
        # Provide multiple values for repeated calls
        mock_time.perf_counter.side_effect = [0.0, 0.05, 0.1, 0.15]
        result = m.bench_copy(1024, duration=0.1)
        assert isinstance(result, (int, float, type(None)))


def test_main_structure():
    """Test main function returns expected structure."""
    m = get_module()
    with mock.patch.object(m, "_DUR", 0.05):
        result = m.main()
    assert isinstance(result, dict)
    assert result.get("type") == "result"
    assert result.get("test") == "memory"
