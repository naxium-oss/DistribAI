"""
DistribAI Benchmark: Path Tracing (GPU + CPU)
============================================
Physically-based path tracer: Lambertian diffuse, multiple bounces,
Russian-roulette termination, emissive spheres.
GPU backend : PyTorch CUDA  (skipped gracefully if unavailable)
CPU backend : NumPy
Score calibration
-----------------
Scores use a log-scale between SCORE_FLOOR (→ 0) and SCORE_CEIL (→ 100).
Override via env vars:
  PT_GPU_FLOOR_MRAYS   (default 1)
  PT_GPU_CEIL_MRAYS    (default 10000)
  PT_CPU_FLOOR_MRAYS   (default 0.1)
  PT_CPU_CEIL_MRAYS    (default 1000)
  PT_GPU_DURATION_S    (default 50)
  PT_CPU_DURATION_S    (default 35)
"""

import json
import math
import os
import shutil
import subprocess
import time

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import torch
    import torch.nn.functional as functional

    _HAS_TORCH = True
    _HAS_CUDA = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = _HAS_CUDA = False
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
_GPU_FLOOR = float(os.environ.get("PT_GPU_FLOOR_MRAYS", 1))
_GPU_CEIL = float(os.environ.get("PT_GPU_CEIL_MRAYS", 10000))
_CPU_FLOOR = float(os.environ.get("PT_CPU_FLOOR_MRAYS", 0.1))
_CPU_CEIL = float(os.environ.get("PT_CPU_CEIL_MRAYS", 1000))
_GPU_DUR = float(os.environ.get("PT_GPU_DURATION_S", 10))
_CPU_DUR = float(os.environ.get("PT_CPU_DURATION_S", 7))


def emit(data: dict):
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    """
    Log-normalised score in [0, 100].
    floor → ~0    ceil → ~100
    """
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def _mean(values: list[float]) -> float:
    """Calculate mean with fallback if numpy not available."""
    if not values:
        return 0.0
    if HAS_NUMPY:
        return float(np.mean(values))
    return sum(values) / len(values)


def get_gpu_stats() -> dict | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=temperature.gpu,clocks.current.graphics,clocks.max.graphics,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        return {
            "temp": float(parts[0]),
            "cur_clk": float(parts[1]),
            "max_clk": float(parts[2]),
            "power": float(parts[3]),
        }
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


_SPHERE_GEOM = [
    [0.0, -100.5, -3.0, 100.0],
    [0.0, 3.5, -3.0, 1.0],
    [-3.0, 0.0, -3.5, 1.8],
    [3.0, 0.0, -3.5, 1.8],
    [0.0, 0.0, -3.0, 0.8],
    [-0.9, -0.25, -1.8, 0.25],
    [0.9, -0.25, -1.8, 0.25],
    [0.0, -0.30, -1.3, 0.18],
]
_SPHERE_ALBEDO = [
    [0.82, 0.82, 0.82],
    [1.00, 0.97, 0.88],
    [0.80, 0.12, 0.10],
    [0.10, 0.12, 0.80],
    [0.92, 0.92, 0.92],
    [0.92, 0.52, 0.10],
    [0.15, 0.80, 0.20],
    [0.92, 0.88, 0.10],
]
_SPHERE_EMISSION = [0, 14, 0, 0, 0, 0, 0, 0]
_CAM_POS = [0.0, 0.5, 2.5]
_CAM_FOV = math.pi / 3.0


def _torch_intersect(ray_o, ray_d, sph_c, sph_r):
    """
    ray_o  [N,3]  ray_d  [N,3] (unit)
    sph_c  [S,3]  sph_r  [S]
    → t    [N,S]  (inf = no hit)
    """
    s = sph_c.shape[0]
    oc = ray_o.unsqueeze(1) - sph_c.unsqueeze(0)
    rd = ray_d.unsqueeze(1).expand(-1, s, -1)
    a = (rd * rd).sum(-1)
    b = 2.0 * (oc * rd).sum(-1)
    c = (oc * oc).sum(-1) - sph_r.unsqueeze(0) ** 2
    dis = b * b - 4 * a * c
    ok = dis >= 0
    sq = torch.sqrt(dis.clamp(min=0))
    inf = torch.full_like(dis, float("inf"))
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    t = torch.where(ok & (t1 > 1e-4), t1, torch.where(ok & (t2 > 1e-4), t2, inf))
    return t


def _gpu_render_frame(device, sph_c, sph_r, albedo, emission, w: int, h: int, bounces: int) -> int:
    n = w * h
    half = math.tan(_CAM_FOV / 2)
    asp = w / h
    g_i = torch.arange(w, device=device, dtype=torch.float32)
    g_j = torch.arange(h, device=device, dtype=torch.float32)
    jj, ii = torch.meshgrid(g_j, g_i, indexing="ij")
    ii, jj = ii.reshape(-1), jj.reshape(-1)
    jx = torch.rand(n, device=device) - 0.5
    jy = torch.rand(n, device=device) - 0.5
    dx = ((ii + jx) / w * 2 - 1) * half * asp
    dy = (1 - (jj + jy) / h * 2) * half
    dz = torch.full((n,), -1.0, device=device)
    dirs = functional.normalize(torch.stack([dx, dy, dz], 1), dim=1)
    origins = (
        torch.tensor(_CAM_POS, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(n, -1)
        .clone()
    )
    radiance = torch.zeros(n, 3, device=device)
    beta = torch.ones(n, 3, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    total_ops = 0
    for b_i in range(bounces):
        n_alive = int(alive.sum())
        if n_alive == 0:
            break
        total_ops += n_alive
        a_idx = alive.nonzero(as_tuple=True)[0]
        t = _torch_intersect(origins[a_idx], dirs[a_idx], sph_c, sph_r)
        t_min, hit_sph = t.min(dim=1)
        hit = t_min < 1e9
        m_idx = a_idx[~hit]
        if m_idx.numel():
            sky_y = dirs[m_idx, 1].clamp(0, 1).unsqueeze(1)
            sky_col = sky_y * dirs.new_tensor([[0.04, 0.04, 0.06]]) + (1 - sky_y) * dirs.new_tensor(
                [[0.08, 0.08, 0.14]]
            )
            radiance[m_idx] += beta[m_idx] * sky_col
            alive[m_idx] = False
        h_idx = a_idx[hit]
        if not h_idx.numel():
            continue
        ht = t_min[hit].unsqueeze(1)
        ho = origins[h_idx]
        hd = dirs[h_idx]
        hs = hit_sph[hit]
        pt = ho + ht * hd
        n_vec = functional.normalize(pt - sph_c[hs], dim=1)
        n_vec = torch.where(((hd * n_vec).sum(1, keepdim=True) < 0).expand_as(n_vec), n_vec, -n_vec)
        radiance[h_idx] += beta[h_idx] * albedo[hs] * emission[hs].unsqueeze(1)
        rand = functional.normalize(n_vec + torch.randn_like(n_vec), dim=1)
        cos_val = (rand * n_vec).sum(1, keepdim=True).clamp(0, 1)
        rand = torch.where(cos_val < 0, -rand, rand)
        cos_val = cos_val.abs()
        beta[h_idx] = beta[h_idx] * albedo[hs] * cos_val * 2.0
        if b_i >= 2:
            rr = beta[h_idx].max(1).values.clamp(0, 0.95)
            dead = torch.rand(h_idx.shape[0], device=device) > rr
            alive[h_idx[dead]] = False
            keep = ~dead
            if keep.any():
                beta[h_idx[keep]] /= rr[keep].unsqueeze(1).clamp(min=1e-7)
        origins[h_idx] = pt + n_vec * 1e-3
        dirs[h_idx] = rand
    return total_ops


def benchmark_gpu():
    if not _HAS_CUDA:
        emit({"type": "skip", "test": "pathtracing_gpu", "reason": "CUDA not available"})
        return None
    device = torch.device("cuda")
    emit(
        {
            "type": "progress",
            "test": "pathtracing_gpu",
            "pct": 0,
            "message": "Warming up GPU path tracer…",
        }
    )
    sph_c = torch.tensor(_SPHERE_GEOM, dtype=torch.float32, device=device)[:, :3]
    sph_r = torch.tensor(_SPHERE_GEOM, dtype=torch.float32, device=device)[:, 3]
    albedo = torch.tensor(_SPHERE_ALBEDO, dtype=torch.float32, device=device)
    emission = torch.tensor(_SPHERE_EMISSION, dtype=torch.float32, device=device)
    free_bytes = torch.cuda.get_device_properties(device).total_memory
    if free_bytes >= 16 * 2**30:
        w, h = 512, 512
    elif free_bytes >= 6 * 2**30:
        w, h = 256, 256
    else:
        w, h = 128, 128
    bounces = 5
    emit(
        {
            "type": "progress",
            "test": "pathtracing_gpu",
            "pct": 1,
            "message": "Compiling CUDA kernels (first run only, may take a moment)…",
        }
    )
    _gpu_render_frame(device, sph_c, sph_r, albedo, emission, 32, 32, 1)
    torch.cuda.synchronize()
    emit(
        {
            "type": "progress",
            "test": "pathtracing_gpu",
            "pct": 3,
            "message": f"Kernels compiled — running full warmup at {w}×{h}...",
        }
    )
    _gpu_render_frame(device, sph_c, sph_r, albedo, emission, w, h, bounces)
    torch.cuda.synchronize()
    throughputs: list[float] = []
    temps: list[float] = []
    total_ops = 0
    t0 = time.perf_counter()
    deadline = t0 + _GPU_DUR
    frame = 0
    while time.perf_counter() < deadline:
        f0 = time.perf_counter()
        ops = _gpu_render_frame(device, sph_c, sph_r, albedo, emission, w, h, bounces)
        torch.cuda.synchronize()
        dt = time.perf_counter() - f0
        tp = ops / dt / 1e6
        throughputs.append(tp)
        total_ops += ops
        frame += 1
        stats = get_gpu_stats()
        if stats and "temp" in stats:
            temps.append(stats["temp"])
        elapsed = time.perf_counter() - t0
        avg_tp = float(_mean(throughputs[-5:]) if len(throughputs) >= 5 else _mean(throughputs))
        emit(
            {
                "type": "progress",
                "test": "pathtracing_gpu",
                "pct": round(min(99.0, elapsed / _GPU_DUR * 100), 1),
                "message": f"GPU: {avg_tp:.0f} MRays/s | frame {frame} | {total_ops / 1e9:.3f}B ray-ops",
                "throughput_mrays": round(avg_tp, 1),
                "temp": stats["temp"] if stats else None,
                "elapsed": round(elapsed, 1),
            }
        )
    avg_tp = float(_mean(throughputs))
    early_tp = float(_mean(throughputs[: max(1, len(throughputs) // 3)]))
    late_tp = float(_mean(throughputs[max(0, len(throughputs) * 2 // 3) :]))
    ratio = late_tp / early_tp if early_tp > 0 else 1.0
    throttled = False
    if temps:
        max_temp = max(temps)
        if max_temp > 85:
            throttled = True
            emit(
                {
                    "type": "thermal_warning",
                    "test": "pathtracing_gpu",
                    "max_temp": max_temp,
                    "message": f"GPU hit {max_temp:.0f}°C — thermal throttling likely. Check airflow / fan curve.",
                }
            )
    if ratio < 0.85:
        throttled = True
        emit(
            {
                "type": "thermal_warning",
                "test": "pathtracing_gpu",
                "drop_pct": round((1 - ratio) * 100),
                "message": f"GPU throughput fell {round((1 - ratio) * 100)}% over test — possible thermal throttling.",
            }
        )
    score = log_score(avg_tp, _GPU_FLOOR, _GPU_CEIL)
    result = {
        "type": "result",
        "test": "pathtracing_gpu",
        "throughput_mrays": round(avg_tp, 1),
        "peak_mrays": round(max(throughputs), 1),
        "total_ray_ops_billion": round(total_ops / 1e9, 3),
        "resolution": f"{w}x{h}",
        "bounces": bounces,
        "frames": frame,
        "score": round(score, 1),
        "thermal_throttled": throttled,
        "max_temp_c": round(max(temps), 1) if temps else None,
    }
    emit(result)
    return result


def _np_intersect(ox, oy, oz, dx, dy, dz, sph_c, sph_r):
    ocx = ox[:, None] - sph_c[:, 0]
    ocy = oy[:, None] - sph_c[:, 1]
    ocz = oz[:, None] - sph_c[:, 2]
    rdx = dx[:, None]
    rdy = dy[:, None]
    rdz = dz[:, None]
    a = rdx * rdx + rdy * rdy + rdz * rdz
    b = 2 * (ocx * rdx + ocy * rdy + ocz * rdz)
    c = ocx * ocx + ocy * ocy + ocz * ocz - sph_r**2
    dis = b * b - 4 * a * c
    ok = dis >= 0
    sq = np.sqrt(np.maximum(dis, 0))
    t1 = np.where(ok, (-b - sq) / (2 * a), np.inf)
    t2 = np.where(ok, (-b + sq) / (2 * a), np.inf)
    t = np.where(ok & (t1 > 1e-4), t1, np.where(ok & (t2 > 1e-4), t2, np.inf))
    return t.astype(np.float32)


def _cpu_render_frame(sph_c, sph_r, albedo, emission, w: int, h: int, bounces: int) -> int:
    n = w * h
    half = math.tan(_CAM_FOV / 2)
    asp = w / h
    grid_i, grid_j = np.meshgrid(np.arange(w), np.arange(h))
    ii = grid_i.reshape(-1).astype(np.float32)
    jj = grid_j.reshape(-1).astype(np.float32)
    jx = (np.random.rand(n) - 0.5).astype(np.float32)
    jy = (np.random.rand(n) - 0.5).astype(np.float32)
    dx = ((ii + jx) / w * 2 - 1) * half * asp
    dy = (1 - (jj + jy) / h * 2) * half
    dz = np.full(n, -1.0, dtype=np.float32)
    nrm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / nrm, dy / nrm, dz / nrm
    cx, cy, cz = float(_CAM_POS[0]), float(_CAM_POS[1]), float(_CAM_POS[2])
    ox = np.full(n, cx, dtype=np.float32)
    oy = np.full(n, cy, dtype=np.float32)
    oz = np.full(n, cz, dtype=np.float32)
    rad = np.zeros((n, 3), dtype=np.float32)
    beta = np.ones((n, 3), dtype=np.float32)
    total_ops = 0
    for _ in range(bounces):
        total_ops += n
        t = _np_intersect(ox, oy, oz, dx, dy, dz, sph_c, sph_r)
        t_min = t.min(axis=1)
        hit_sph = t.argmin(axis=1)
        hit = t_min < 1e9
        miss = ~hit
        if miss.any():
            sky_t = np.clip(dy[miss], 0, 1)[:, None]
            sky = sky_t * np.array([[0.04, 0.04, 0.06]]) + (1 - sky_t) * np.array(
                [[0.08, 0.08, 0.14]]
            )
            rad[miss] += beta[miss] * sky
            beta[miss] = 0.0
        if not hit.any():
            break
        ht = t_min[hit]
        hs = hit_sph[hit]
        ptx = ox[hit] + ht * dx[hit]
        pty = oy[hit] + ht * dy[hit]
        ptz = oz[hit] + ht * dz[hit]
        nx = ptx - sph_c[hs, 0]
        ny = pty - sph_c[hs, 1]
        nz = ptz - sph_c[hs, 2]
        r_ = sph_r[hs].clip(1e-9)
        nx, ny, nz = nx / r_, ny / r_, nz / r_
        dot = dx[hit] * nx + dy[hit] * ny + dz[hit] * nz
        fl = dot < 0
        nx = np.where(fl, nx, -nx)
        ny = np.where(fl, ny, -ny)
        nz = np.where(fl, nz, -nz)
        rad[hit] += beta[hit] * albedo[hs] * emission[hs, np.newaxis]
        rx = nx + np.random.randn(hit.sum()).astype(np.float32)
        ry = ny + np.random.randn(hit.sum()).astype(np.float32)
        rz = nz + np.random.randn(hit.sum()).astype(np.float32)
        rn = np.sqrt(rx * rx + ry * ry + rz * rz).clip(1e-9)
        rx, ry, rz = rx / rn, ry / rn, rz / rn
        cos = np.maximum(rx * nx + ry * ny + rz * nz, 0)[:, None]
        rx = np.where(cos[:, 0] < 0, -rx, rx)
        ry = np.where(cos[:, 0] < 0, -ry, ry)
        rz = np.where(cos[:, 0] < 0, -rz, rz)
        cos = np.abs(cos)
        beta[hit] = beta[hit] * albedo[hs] * cos * 2.0
        ox[hit] = ptx + nx * 1e-3
        oy[hit] = pty + ny * 1e-3
        oz[hit] = ptz + nz * 1e-3
        dx[hit] = rx
        dy[hit] = ry
        dz[hit] = rz
        beta[~hit] = 0.0
    return total_ops


def benchmark_cpu():
    emit(
        {
            "type": "progress",
            "test": "pathtracing_cpu",
            "pct": 0,
            "message": "Starting CPU path tracer (NumPy)…",
        }
    )
    sph_c = np.array(_SPHERE_GEOM, dtype=np.float32)[:, :3]
    sph_r = np.array(_SPHERE_GEOM, dtype=np.float32)[:, 3]
    albedo = np.array(_SPHERE_ALBEDO, dtype=np.float32)
    emission = np.array(_SPHERE_EMISSION, dtype=np.float32)
    w, h = 128, 128
    bounces = 5
    throughputs: list[float] = []
    freq_samples: list[float] = []
    total_ops = 0
    t0 = time.perf_counter()
    deadline = t0 + _CPU_DUR
    frame = 0
    while time.perf_counter() < deadline:
        f0 = time.perf_counter()
        ops = _cpu_render_frame(sph_c, sph_r, albedo, emission, w, h, bounces)
        dt = time.perf_counter() - f0
        tp = ops / dt / 1e6
        throughputs.append(tp)
        total_ops += ops
        frame += 1
        if _HAS_PSUTIL:
            freq = psutil.cpu_freq()
            if freq:
                freq_samples.append(freq.current)
        elapsed = time.perf_counter() - t0
        avg_tp = float(_mean(throughputs[-5:]) if len(throughputs) >= 5 else _mean(throughputs))
        emit(
            {
                "type": "progress",
                "test": "pathtracing_cpu",
                "pct": round(min(99.0, elapsed / _CPU_DUR * 100), 1),
                "message": f"CPU: {avg_tp:.2f} MRays/s | frame {frame} | {total_ops / 1e6:.0f}M ray-ops",
                "throughput_mrays": round(avg_tp, 3),
                "elapsed": round(elapsed, 1),
            }
        )
    avg_tp = float(_mean(throughputs))
    throttled = False
    if freq_samples and len(freq_samples) >= 12:
        n = len(freq_samples)
        mid_f = float(_mean(freq_samples[n // 3 : n * 2 // 3]))
        late_f = float(_mean(freq_samples[-max(1, n // 4) :]))
        if mid_f > 0 and late_f / mid_f < 0.80:
            throttled = True
            emit(
                {
                    "type": "thermal_warning",
                    "test": "pathtracing_cpu",
                    "drop_pct": round((1 - late_f / mid_f) * 100),
                    "message": f"CPU clock dropped {round((1 - late_f / mid_f) * 100)}% (mid-run vs end) — thermal throttling detected.",
                }
            )
    score = log_score(avg_tp, _CPU_FLOOR, _CPU_CEIL)
    result = {
        "type": "result",
        "test": "pathtracing_cpu",
        "throughput_mrays": round(avg_tp, 3),
        "peak_mrays": round(max(throughputs), 3),
        "total_ray_ops_million": round(total_ops / 1e6, 1),
        "resolution": f"{w}x{h}",
        "bounces": bounces,
        "frames": frame,
        "score": round(score, 1),
        "thermal_throttled": throttled,
    }
    emit(result)
    return result


def main():
    emit(
        {
            "type": "benchmark_group_start",
            "name": "pathtracing",
            "display": "Path Tracing",
            "message": "Real path tracer: Lambertian diffuse, 5 bounces, 8-sphere Cornell-box scene.",
        }
    )
    gpu_res = benchmark_gpu()
    cpu_res = benchmark_cpu()
    gpu_score = gpu_res.get("score", 0) if gpu_res else 0.0
    cpu_score = cpu_res.get("score", 0) if cpu_res else 0.0
    combined = max(gpu_score, cpu_score)
    throttled = bool(
        (gpu_res and gpu_res.get("thermal_throttled"))
        or (cpu_res and cpu_res.get("thermal_throttled"))
    )
    emit(
        {
            "type": "result",
            "test": "pathtracing",
            "gpu_score": round(gpu_score, 1),
            "cpu_score": round(cpu_score, 1),
            "score": round(combined, 1),
            "gpu_mrays": gpu_res.get("throughput_mrays", 0) if gpu_res else 0,
            "cpu_mrays": cpu_res.get("throughput_mrays", 0) if cpu_res else 0,
            "thermal_throttled": throttled,
        }
    )


if __name__ == "__main__":
    main()
