"""Simple tests for benchmark modules to increase coverage."""

import json


def test_bench_memory_emit():
    """Test bench_memory emit function."""
    import io
    import sys

    from worker.src.benchmark.bench_memory import emit

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    test_data = {"type": "test", "value": 42}
    emit(test_data)

    output = captured.getvalue()
    sys.stdout = old_stdout

    parsed = json.loads(output.strip())
    assert parsed == test_data


def test_bench_memory_log_score():
    """Test bench_memory log_score function."""
    from worker.src.benchmark.bench_memory import log_score

    # Test with valid values
    score = log_score(100.0, 10.0, 1000.0)
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0

    # Test with None
    score = log_score(None, 10.0, 1000.0)
    assert score == 0.0

    # Test with invalid floor/ceil
    score = log_score(100.0, 0.0, 1000.0)
    assert score == 0.0


def test_bench_memory_choose_array_bytes():
    """Test _choose_array_bytes function."""
    from worker.src.benchmark.bench_memory import _choose_array_bytes

    result = _choose_array_bytes()
    assert isinstance(result, int)
    assert result > 0


def test_bench_memory_fmt():
    """Test _fmt function."""
    from worker.src.benchmark.bench_memory import _fmt

    # Test with value
    result = _fmt(100.0, ".2f")
    assert isinstance(result, str)

    # Test with None
    result = _fmt(None, ".2f", "N/A")
    assert result == "N/A"


def test_bench_memory_constants():
    """Test bench_memory constants."""
    from worker.src.benchmark.bench_memory import _CEIL, _DUR, _FLOOR

    assert isinstance(_FLOOR, float)
    assert isinstance(_CEIL, float)
    assert isinstance(_DUR, float)
    assert _FLOOR > 0
    assert _CEIL > _FLOOR
    assert _DUR > 0


def test_bench_vram_emit():
    """Test bench_vram emit function."""
    import io
    import sys

    from worker.src.benchmark.bench_vram import emit

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    test_data = {"type": "vram_test"}
    emit(test_data)

    output = captured.getvalue()
    sys.stdout = old_stdout

    parsed = json.loads(output.strip())
    assert parsed == test_data


def test_bench_vram_has_constants():
    """Test bench_vram has expected attributes."""
    import worker.src.benchmark.bench_vram as bench_vram

    # Should have expected functions
    assert hasattr(bench_vram, "emit")
    assert callable(bench_vram.emit)


def test_bench_network_emit():
    """Test bench_network emit function."""
    import io
    import sys

    from worker.src.benchmark.bench_network import emit

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    test_data = {"type": "network_test"}
    emit(test_data)

    output = captured.getvalue()
    sys.stdout = old_stdout

    parsed = json.loads(output.strip())
    assert parsed == test_data


def test_bench_runner_emit():
    """Test bench_runner emit function."""
    import io
    import sys

    from worker.src.benchmark.bench_runner import emit

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    test_data = {"type": "runner_test"}
    emit(test_data)

    output = captured.getvalue()
    sys.stdout = old_stdout

    parsed = json.loads(output.strip())
    assert parsed == test_data


def test_bench_tensor_emit():
    """Test bench_tensor emit function."""
    import io
    import sys

    from worker.src.benchmark.bench_tensor import emit

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    test_data = {"type": "tensor_test"}
    emit(test_data)

    output = captured.getvalue()
    sys.stdout = old_stdout

    parsed = json.loads(output.strip())
    assert parsed == test_data


def test_bench_pathtracing_emit():
    """Test bench_pathtracing emit function."""
    import io
    import sys

    from worker.src.benchmark.bench_pathtracing import emit

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    test_data = {"type": "pathtracing_test"}
    emit(test_data)

    output = captured.getvalue()
    sys.stdout = old_stdout

    parsed = json.loads(output.strip())
    assert parsed == test_data
