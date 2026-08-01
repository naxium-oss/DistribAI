"""
DistribAI Benchmark Runner
==========================

Orchestrates all benchmark sub-tests, aggregates results, computes an
overall score, and recommends a hardware tier + suitable task types.

Usage:
    python3 bench_runner.py [--skip <name,...>] [--only <name,...>]

Each sub-benchmark is a standalone Python script in the same directory.
Results are written as JSON lines to stdout — the dashboard server streams
these directly to the browser via SSE.
"""

from __future__ import annotations

"""
Overall score formula
---------------------
overall = weighted average of available individual scores, where weights are:
  tensor      : 0.35   (primary compute — directly affects training speed)
  pathtracing : 0.20   (GPU/CPU parallel compute)
  vram        : 0.15   (determines model size eligibility)
  memory      : 0.10   (RAM bandwidth)
  network     : 0.10   (upload/download matters for gradient transfer)
  write       : 0.10   (dataset I/O)

Missing tests (skipped/errored) are excluded and weights re-normalised.

Hardware tier assignment
------------------------
Tiers are assigned based on the *combination* of scores, not a single number.
The recommended task types come directly from DistribAI's load-balancer spec.
"""
import argparse
import json
import os
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BENCHMARKS = [
    ("pathtracing", "bench_pathtracing.py", "Path Tracing (GPU + CPU)"),
    ("memory", "bench_memory.py", "Memory Bandwidth"),
    ("network", "bench_network.py", "Network Throughput"),
    ("write", "bench_write.py", "Disk I/O"),
    ("tensor", "bench_tensor.py", "Tensor Compute"),
    ("vram", "bench_vram.py", "VRAM Capacity"),
    ("ml_workloads", "bench_ml_workloads.py", "ML Workloads"),
    ("distributed_training", "bench_distributed_training.py", "Distributed Training"),
    ("data_pipeline", "bench_data_pipeline.py", "Data Pipeline"),
    ("system_stability", "bench_system_stability.py", "System Stability"),
    ("memory_efficiency", "bench_memory_efficiency.py", "Memory Efficiency"),
    ("io_patterns", "bench_io_patterns.py", "I/O Patterns"),
    ("cpu_gpu_bandwidth", "bench_cpu_gpu_bandwidth.py", "CPU-GPU Bandwidth"),
    ("parallel_processing", "bench_parallel_processing.py", "Parallel Processing"),
]
# Default suite matches the contributor dashboard cards (fast + complete scores).
_CORE_BENCHMARKS = {
    "pathtracing",
    "memory",
    "network",
    "write",
    "tensor",
    "vram",
}
_WEIGHTS = {
    "ml_workloads": 0.08,
    "distributed_training": 0.08,
    "data_pipeline": 0.06,
    "system_stability": 0.06,
    "memory_efficiency": 0.04,
    "io_patterns": 0.04,
    "cpu_gpu_bandwidth": 0.04,
    "parallel_processing": 0.04,
    "tensor": 0.20,
    "pathtracing": 0.12,
    "vram": 0.09,
    "memory": 0.06,
    "network": 0.06,
    "write": 0.03,  # Reduced from 0.06 to make total sum to 1.0
}


def estimate_throughput(score: float, metric_type: str) -> str:
    """
    Estimate throughput based on benchmark score with contextual meaning.
    Returns human-readable throughput estimates.
    """
    if score <= 0:
        return "N/A"

    # Base throughput calculations (these are calibrated estimates)
    if metric_type == "samples":
        # ML workload samples per second
        base_throughput = 100  # Base samples/sec at score 50
        if score > 50:
            return f"{int(base_throughput * (score / 50))} samples/sec"
        else:
            return f"{int(base_throughput * (score / 50))} samples/sec"

    elif metric_type == "tokens":
        # Token processing per second
        base_throughput = 1000  # Base tokens/sec at score 50
        return f"{int(base_throughput * (score / 50))} tokens/sec"

    elif metric_type == "mb":
        # Data throughput in MB/sec
        base_throughput = 100  # Base MB/sec at score 50
        return f"{int(base_throughput * (score / 50))} MB/sec"

    elif metric_type == "ops":
        # Operations per second
        base_throughput = 1000000  # Base ops/sec at score 50
        return f"{int(base_throughput * (score / 50))} ops/sec"

    return "N/A"


def get_avoid_workloads(tier_info: dict) -> list[str]:
    """Get workloads to avoid based on tier and capabilities."""
    tier = tier_info.get("tier", "")
    constraints = tier_info.get("constraints", {})

    avoid_workloads = []

    if not constraints.get("requires_gpu"):
        avoid_workloads.extend(
            ["GPU-intensive training", "Large model fine-tuning", "CUDA-accelerated inference"]
        )

    if tier in ["Auxiliary Node", "CPU Processing Node"]:
        avoid_workloads.extend(
            ["Large-batch training", "Multi-GPU coordination", "High-throughput inference"]
        )

    if constraints.get("max_model_size_gb", 0) < 4:
        avoid_workloads.extend(
            [
                "Large model training (>4GB)",
                "High-resolution image processing",
                "Long sequence transformers",
            ]
        )

    return avoid_workloads


def get_optimization_tips(results: dict[str, dict]) -> list[str]:
    """Get optimization tips based on benchmark results."""
    tips = []

    # Check for low scores and provide tips
    for name, result in results.items():
        score = result.get("score", 0)

        if name == "memory" and score < 30:
            tips.append("Consider increasing RAM or optimizing memory usage for better performance")

        if name == "network" and score < 30:
            tips.append("Network performance may limit distributed training effectiveness")

        if name == "vram" and score < 30:
            tips.append("VRAM is limited - consider using smaller batch sizes or model parallelism")

        if name == "tensor" and score < 30:
            tips.append("GPU compute performance is low - check GPU drivers and thermal management")

        if name == "system_stability" and score < 40:
            tips.append("System stability issues detected - check cooling and power management")

        if name == "data_pipeline" and score < 30:
            tips.append(
                "Data pipeline performance may bottleneck training - consider faster storage"
            )

        if name == "distributed_training" and score < 30:
            tips.append("Distributed training performance is low - check network configuration")

        if name == "parallel_processing" and score < 30:
            tips.append("Parallel processing efficiency is low - check CPU core utilization")

    return tips


def emit(data: dict):
    print(json.dumps(data), flush=True)


def run_benchmark(name: str, script: str, display: str) -> dict | None:
    emit(
        {
            "type": "benchmark_start",
            "name": name,
            "display": display,
            "message": f"Starting: {display}…",
        }
    )
    script_path = os.path.join(_SCRIPT_DIR, script)
    if not os.path.isfile(script_path):
        emit({"type": "error", "name": name, "message": f"Script not found: {script_path}"})
        return None
    final_result = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Normalize progress so the dashboard always finds a test key.
                if data.get("type") in ("progress", "test_progress") and not data.get("test"):
                    data["test"] = name
                emit(data)
                if data.get("type") == "result" and data.get("test") in (name, None):
                    if data.get("test") is None:
                        data["test"] = name
                    final_result = data
            except json.JSONDecodeError:
                emit({"type": "log", "test": name, "message": line[:400]})
        # Drain stderr without blocking the next suite step for long.
        err_tail = ""
        if proc.stderr is not None:
            try:
                err_tail = (proc.stderr.read(2000) or "").strip()
            except OSError:
                err_tail = ""
        proc.wait(timeout=900)
        if proc.returncode != 0:
            emit(
                {
                    "type": "error",
                    "name": name,
                    "test": name,
                    "message": f"Exited with code {proc.returncode}: {err_tail}",
                }
            )
    except subprocess.TimeoutExpired:
        proc.kill()
        emit({"type": "error", "name": name, "message": "Benchmark timed out (15 min)."})
    except Exception as exc:
        emit({"type": "error", "name": name, "message": str(exc)})
    throttled = bool(final_result and final_result.get("thermal_throttled"))
    emit(
        {
            "type": "benchmark_end",
            "name": name,
            "display": display,
            "score": final_result.get("score", 0) if final_result else 0,
            "throttled": throttled,
        }
    )
    return final_result


def compute_overall(results: dict[str, dict]) -> float:
    scores = {name: r.get("score", 0.0) for name, r in results.items() if r}
    weights = {k: _WEIGHTS[k] for k in scores if k in _WEIGHTS}
    if not weights:
        return 0.0
    total_w = sum(weights.values())
    weighted = sum(scores.get(k, 0) * weights[k] for k in weights)
    return min(100.0, weighted / total_w)


def assign_tier(overall: float, results: dict[str, dict]) -> dict:
    """
    Determine hardware tier and recommended task types based on measured scores.
    Provides contextual meaning for distributed training task distribution.
    """
    gpu_used = bool(results.get("tensor", {}) and results["tensor"].get("gpu_used"))
    vram_gb = float(results.get("vram", {}).get("max_usable_gb", 0) if results.get("vram") else 0)
    net_s = float(results.get("network", {}).get("score", 0) if results.get("network") else 0)

    # Extract individual benchmark scores for detailed analysis
    ml_score = float(
        results.get("ml_workloads", {}).get("score", 0) if results.get("ml_workloads") else 0
    )
    distributed_score = float(
        results.get("distributed_training", {}).get("score", 0)
        if results.get("distributed_training")
        else 0
    )
    data_score = float(
        results.get("data_pipeline", {}).get("score", 0) if results.get("data_pipeline") else 0
    )
    stability_score = float(
        results.get("system_stability", {}).get("score", 0)
        if results.get("system_stability")
        else 0
    )
    memory_score = float(
        results.get("memory_efficiency", {}).get("score", 0)
        if results.get("memory_efficiency")
        else 0
    )
    io_score = float(
        results.get("io_patterns", {}).get("score", 0) if results.get("io_patterns") else 0
    )
    bandwidth_score = float(
        results.get("cpu_gpu_bandwidth", {}).get("score", 0)
        if results.get("cpu_gpu_bandwidth")
        else 0
    )
    parallel_score = float(
        results.get("parallel_processing", {}).get("score", 0)
        if results.get("parallel_processing")
        else 0
    )

    # Determine task capacity and constraints
    task_capacity = {
        "max_model_size_gb": vram_gb,
        "batch_size_capability": "large"
        if ml_score > 70
        else "medium"
        if ml_score > 40
        else "small",
        "distributed_efficiency": "high"
        if distributed_score > 70
        else "medium"
        if distributed_score > 40
        else "low",
        "data_throughput": "high" if data_score > 70 else "medium" if data_score > 40 else "low",
        "stability_rating": "excellent"
        if stability_score > 80
        else "good"
        if stability_score > 60
        else "fair",
        "memory_efficiency": "high"
        if memory_score > 70
        else "medium"
        if memory_score > 40
        else "low",
        "io_performance": "high" if io_score > 70 else "medium" if io_score > 40 else "low",
        "bandwidth_capability": "high"
        if bandwidth_score > 70
        else "medium"
        if bandwidth_score > 40
        else "low",
        "parallel_capability": "high"
        if parallel_score > 70
        else "medium"
        if parallel_score > 40
        else "low",
    }

    # Calculate specialization scores
    specialization = {
        "training": (ml_score + distributed_score + memory_score) / 3,
        "inference": (ml_score + memory_score + bandwidth_score) / 3,
        "data_processing": (data_score + io_score + parallel_score) / 3,
        "preprocessing": (data_score + memory_score + io_score) / 3,
    }

    # Determine primary specialization
    primary_specialization = max(specialization, key=specialization.get)

    # Generate throughput estimates
    throughput_estimates = {
        "samples_per_second": estimate_throughput(ml_score, "samples"),
        "tokens_per_second": estimate_throughput(ml_score, "tokens"),
        "mb_per_second": estimate_throughput(io_score, "mb"),
        "ops_per_second": estimate_throughput(ml_score, "ops"),
    }

    if gpu_used and vram_gb >= 8 and overall >= 65:
        tier = "Flagship Training Node"
        tasks = [
            "Full model pre-training & fine-tuning (P1–P2)",
            "Large-batch gradient accumulation tasks",
            "Transformer architecture training",
            "Multi-GPU distributed training coordination",
            "High-throughput inference serving",
        ]
        blurb = (
            "Your hardware qualifies for the most demanding training tasks. "
            f"With {vram_gb:.1f} GB usable VRAM and strong compute scores, "
            "you can handle large models and high-throughput gradient steps. "
            f"Primary specialization: {primary_specialization} ({specialization[primary_specialization]:.1f}/100)."
        )
        max_tp = "High"
    elif gpu_used and overall >= 45:
        tier = "Standard Training Node"
        tasks = [
            "P2–P3 micro-task training",
            "Small-to-medium model fine-tuning",
            "Gradient accumulation (moderate batch sizes)",
        ]
        blurb = (
            "A solid performer for most distributed training tasks in DistribAI. "
            "You'll be assigned fine-tuning and gradient steps on models that fit your VRAM."
        )
        max_tp = "Medium–High"
    elif gpu_used and overall >= 25:
        tier = "Light Training Node"
        tasks = [
            "P3 micro-tasks",
            "Small model fine-tuning",
            "Inference-only evaluation tasks",
        ]
        blurb = (
            "Good for lightweight training tasks. Your node will be assigned shorter, "
            "lower-memory micro-tasks to maximise utilisation."
        )
        max_tp = "Medium"
    elif net_s >= 40 and not gpu_used:
        tier = "Data Pipeline Node"
        tasks = [
            "Dataset tokenisation & preprocessing",
            "High-throughput data loading",
            "Gradient serialisation pass",
        ]
        blurb = (
            "Strong network bandwidth makes you ideal for data pipeline tasks. "
            "Your node will focus on preprocessing and data transfer rather than compute."
        )
        max_tp = "Medium (network-bound)"
    elif overall >= 15:
        tier = "CPU Processing Node"
        tasks = [
            "Tokenisation",
            "Evaluation / validation tasks",
            "Data preprocessing",
        ]
        blurb = (
            "Best suited for CPU-bound preprocessing and evaluation tasks per the "
            "DistribAI spec (§2.3). Compute-heavy training tasks require a CUDA-capable GPU."
        )
        max_tp = "Low–Medium"
    else:
        tier = "Auxiliary Node"
        tasks = [
            "Data loading",
            "Simple preprocessing",
        ]
        blurb = (
            "Limited measured compute. Your node will receive auxiliary data tasks. "
            "Consider upgrading hardware or checking driver/thermal configuration."
        )
        max_tp = "Low"
    return {
        "tier": tier,
        "tasks": tasks,
        "description": blurb,
        "max_throughput": max_tp,
        "task_capacity": task_capacity,
        "specialization": specialization,
        "primary_specialization": primary_specialization,
        "throughput_estimates": throughput_estimates,
        "constraints": {
            "max_model_size_gb": vram_gb,
            "requires_gpu": gpu_used,
            "min_network_score": 30 if "distributed_training" in tasks else 0,
            "stability_requirement": "high"
            if tier in ["Flagship Training Node", "Standard Training Node"]
            else "medium",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="DistribAI benchmark runner")
    parser.add_argument("--skip", default="", help="Comma-separated benchmark names to skip")
    parser.add_argument("--only", default="", help="Run only these comma-separated names")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run extended suite (ML/distributed/io extras). Default is the 6 core tests.",
    )
    args = parser.parse_args()
    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
    only_set = {s.strip() for s in args.only.split(",") if s.strip()}
    if not only_set and not args.full:
        only_set = set(_CORE_BENCHMARKS)
    benchmarks = [
        (name, script, display)
        for name, script, display in _BENCHMARKS
        if name not in skip_set and (not only_set or name in only_set)
    ]
    emit(
        {
            "type": "suite_start",
            "total": len(benchmarks),
            "names": [b[0] for b in benchmarks],
            "message": f"DistribAI benchmark suite starting — {len(benchmarks)} tests",
            "timestamp": time.time(),
        }
    )
    all_results: dict[str, dict] = {}
    thermal_flags: list[str] = []
    for i, (name, script, display) in enumerate(benchmarks):
        emit(
            {
                "type": "suite_progress",
                "current": i,
                "total": len(benchmarks),
                "current_name": display,
            }
        )
        result = run_benchmark(name, script, display)
        if result:
            all_results[name] = result
        if result and result.get("thermal_throttled"):
            thermal_flags.append(display)
    overall = compute_overall(all_results)
    tier = assign_tier(overall, all_results)
    individual_scores = {name: round(r.get("score", 0), 1) for name, r in all_results.items()}
    emit(
        {
            "type": "suite_complete",
            "overall_score": round(overall, 1),
            "individual_scores": individual_scores,
            "tier": tier,
            "thermal_warnings": thermal_flags,
            "results": all_results,
            "timestamp": time.time(),
            "contextual_info": {
                "task_capacity": tier.get("task_capacity", {}),
                "specialization": tier.get("specialization", {}),
                "primary_specialization": tier.get("primary_specialization", "general"),
                "throughput_estimates": tier.get("throughput_estimates", {}),
                "constraints": tier.get("constraints", {}),
                "recommendations": {
                    "best_workloads": tier.get("tasks", []),
                    "avoid_workloads": get_avoid_workloads(tier),
                    "optimization_tips": get_optimization_tips(all_results),
                },
            },
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
