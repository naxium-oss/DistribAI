"""
DistribAI Benchmark: System Stability
====================================
Measures thermal throttling detection, sustained performance, and power efficiency.
These metrics determine how reliable a node is under sustained training loads.
"""

import json
import math
import os
import time

_FLOOR = float(os.environ.get("SCORE_FLOOR_MBPS", 1.0))
_CEIL = float(os.environ.get("SCORE_CEIL_MBPS", 100.0))

try:
    import numpy as np
    import psutil
    import torch

    _HAS_TORCH = True
    _HAS_PSUTIL = True
    _HAS_NUMPY = True
except ImportError:
    torch = None
    psutil = None
    np = None
    _HAS_TORCH = _HAS_PSUTIL = _HAS_NUMPY = False


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def get_system_temperature():
    """Get system temperature if available."""
    if not _HAS_PSUTIL:
        return None

    try:
        # Try to get temperature sensors
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                # Look for common temperature sensor names
                for name, entries in temps.items():
                    if any(keyword in name.lower() for keyword in ["cpu", "core", "gpu", "nvidia"]):
                        if entries:
                            return entries[0].current
        return None
    except Exception:
        return None


def get_power_consumption():
    """Get power consumption if available."""
    if not _HAS_PSUTIL:
        return None

    try:
        if hasattr(psutil, "sensors_power"):
            power = psutil.sensors_power()
            if power:
                for _name, entries in power.items():
                    if entries:
                        return entries[0].current
        return None
    except Exception:
        return None


def benchmark_sustained_performance():
    """Benchmark sustained performance under load."""
    if not _HAS_TORCH:
        emit({"type": "skip", "test": "system_stability", "reason": "PyTorch not available"})
        return 0.0

    _ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a compute-intensive task
    if torch.cuda.is_available():
        matrix_size = 1024
        a = torch.randn(matrix_size, matrix_size).cuda()
        b = torch.randn(matrix_size, matrix_size).cuda()
    else:
        matrix_size = 512
        a = torch.randn(matrix_size, matrix_size)
        b = torch.randn(matrix_size, matrix_size)

    # Run sustained computation
    duration = 30  # 30 seconds
    interval = 2  # Check every 2 seconds
    iterations_per_interval = 10

    performance_scores = []
    temperatures = []
    start_time = time.time()

    for i in range(0, duration, interval):
        interval_start = time.time()

        # Perform computation
        for _ in range(iterations_per_interval):
            torch.mm(a, b)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        interval_end = time.time()

        # Calculate performance (operations per second)
        ops_per_sec = (iterations_per_interval * matrix_size * matrix_size * 2) / (
            interval_end - interval_start
        )
        performance_scores.append(ops_per_sec)

        # Check temperature
        temp = get_system_temperature()
        if temp is not None:
            temperatures.append(temp)

        # Calculate progress
        progress = int((i + interval) / duration * 80)
        emit(
            {
                "type": "progress",
                "test": "system_stability",
                "pct": progress,
                "message": f"Sustained performance: {ops_per_sec:.0f} ops/sec, Temp: {temp if temp else 'N/A'}°C",
            }
        )

        # Check if we're approaching the end
        if time.time() - start_time >= duration:
            break

    # Calculate performance stability
    if len(performance_scores) > 1:
        performance_std = np.std(performance_scores)
        performance_mean = np.mean(performance_scores)
        stability_score = performance_mean / (performance_std + 1e-6)  # Higher is better
    else:
        stability_score = 50.0  # Default score

    # Check for thermal throttling
    thermal_throttled = False
    if len(temperatures) > 1:
        temp_increase = temperatures[-1] - temperatures[0]
        if temp_increase > 10:  # Temperature increased by more than 10°C
            thermal_throttled = True

    return log_score(stability_score, _FLOOR, _CEIL), thermal_throttled


def benchmark_memory_stability():
    """Benchmark memory stability under sustained load."""
    if not _HAS_TORCH:
        return 0.0

    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create memory-intensive task
    tensor_size = 1000000  # 1M elements

    # Run sustained memory operations
    duration = 20  # 20 seconds
    memory_scores = []

    start_time = time.time()

    while time.time() - start_time < duration:
        # Allocate and deallocate memory
        tensors = []
        for _ in range(10):
            if torch.cuda.is_available():
                tensor = torch.randn(tensor_size).cuda()
            else:
                tensor = torch.randn(tensor_size)
            tensors.append(tensor)

        # Perform some computation
        for tensor in tensors:
            torch.sum(tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Clean up
        del tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Memory operation score
        memory_scores.append(tensor_size * 10)

    # Calculate memory stability
    avg_memory_ops = sum(memory_scores) / len(memory_scores)
    return log_score(avg_memory_ops, _FLOOR, _CEIL)


def benchmark_power_efficiency():
    """Benchmark power efficiency if available."""
    if not _HAS_PSUTIL:
        return 0.0

    # Get initial power consumption
    initial_power = get_power_consumption()
    if initial_power is None:
        emit(
            {
                "type": "progress",
                "test": "system_stability",
                "pct": 90,
                "message": "Power monitoring not available",
            }
        )
        return 50.0  # Default score

    _ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run compute task
    if torch.cuda.is_available():
        matrix_size = 512
        a = torch.randn(matrix_size, matrix_size).cuda()
        b = torch.randn(matrix_size, matrix_size).cuda()
    else:
        matrix_size = 256
        a = torch.randn(matrix_size, matrix_size)
        b = torch.randn(matrix_size, matrix_size)

    # Measure power under load
    start_time = time.time()
    duration = 10

    while time.time() - start_time < duration:
        for _ in range(5):
            torch.mm(a, b)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Get final power consumption
    final_power = get_power_consumption()

    if final_power is not None:
        power_increase = final_power - initial_power
        # Calculate efficiency (lower power increase is better for efficiency)
        efficiency_score = max(0, 100 - power_increase * 10)  # Arbitrary scaling
        emit(
            {
                "type": "progress",
                "test": "system_stability",
                "pct": 95,
                "message": f"Power increase: {power_increase:.1f}W",
            }
        )
        return efficiency_score

    return 50.0


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "system_stability",
            "display": "System Stability",
            "message": "Testing thermal throttling, sustained performance, and power efficiency...",
        }
    )

    if not _HAS_TORCH:
        emit({"type": "skip", "test": "system_stability", "reason": "PyTorch not available"})
        return

    # Run sub-benchmarks
    sustained_score, thermal_throttled = benchmark_sustained_performance()
    memory_score = benchmark_memory_stability()
    power_score = benchmark_power_efficiency()

    # Calculate overall score
    scores = [sustained_score, memory_score, power_score]
    valid_scores = [s for s in scores if s > 0]

    if not valid_scores:
        overall_score = 0.0
    else:
        overall_score = sum(valid_scores) / len(valid_scores)

    emit(
        {
            "type": "result",
            "test": "system_stability",
            "score": overall_score,
            "sustained_score": sustained_score,
            "memory_score": memory_score,
            "power_score": power_score,
            "thermal_throttled": thermal_throttled,
        }
    )


if __name__ == "__main__":
    main()
