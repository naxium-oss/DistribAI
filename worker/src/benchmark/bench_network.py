"""
DistribAI Benchmark: Network
Measures download Mbps (short CDN probe), loopback Mbps, and TCP latency.
"""

from __future__ import annotations

import http.client
import json
import math
import os
import socket
import ssl
import threading
import time
import urllib.error
from urllib.parse import urlparse

_FLOOR = float(os.environ.get("SCORE_FLOOR_MBPS", 1.0))
_CEIL = float(os.environ.get("SCORE_CEIL_MBPS", 10_000.0))
_DL_DUR = float(os.environ.get("NET_DURATION_S", 2.5))
_LB_DUR = float(os.environ.get("NET_LOOPBACK_S", 2.0))
# Small, reliable probes only — avoid multi-GB ISO fallbacks that stall the suite.
_DOWNLOAD_URLS = [
    "https://speed.cloudflare.com/__down?bytes=25000000",
    "https://speed.cloudflare.com/__down?bytes=10000000",
]
_LATENCY_HOSTS = [
    ("1.1.1.1", 443),
    ("8.8.8.8", 443),
    ("9.9.9.9", 443),
]


def emit(data: dict) -> None:
    print(json.dumps(data), flush=True)


def log_score(value: float, floor: float, ceil: float) -> float:
    if value <= 0 or floor <= 0 or ceil <= floor:
        return 0.0
    raw = math.log10(max(value, floor)) - math.log10(floor)
    rng = math.log10(ceil) - math.log10(floor)
    return min(100.0, max(0.0, raw / rng * 100.0))


def _try_download(url: str, duration: float) -> float | None:
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            return None
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        conn = http.client.HTTPSConnection(
            parsed.netloc, timeout=6, context=ssl.create_default_context()
        )
        conn.request("GET", path, headers={"User-Agent": "DistribAI-Benchmark/1.0"})
        resp = conn.getresponse()
        try:
            if resp.status != 200:
                return None
            start = time.perf_counter()
            deadline = start + duration
            total = 0
            while time.perf_counter() < deadline:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
            elapsed = time.perf_counter() - start
            if elapsed < 0.4 or total < 32768:
                return None
            return (total * 8) / elapsed / 1e6
        finally:
            conn.close()
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        return None


def bench_download(duration: float) -> tuple[float, str]:
    for index, url in enumerate(_DOWNLOAD_URLS):
        pct = 8 + index * 12
        emit(
            {
                "type": "progress",
                "test": "network",
                "pct": pct,
                "message": f"Download probe {index + 1}/{len(_DOWNLOAD_URLS)}…",
            }
        )
        mbps = _try_download(url, duration)
        if mbps and mbps > 0.1:
            return mbps, url
    return 0.0, "none"


def _loopback_server(port: int, stop_event: threading.Event, buf_size: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    srv.settimeout(5.0)
    try:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while not stop_event.is_set():
            try:
                data = conn.recv(buf_size)
                if not data:
                    break
            except OSError:
                break
        conn.close()
    except OSError:
        return
    finally:
        srv.close()


def bench_loopback(duration: float) -> float:
    buf_size = 256 * 1024
    payload = b"\x00" * buf_size
    with socket.socket() as tmp:
        tmp.bind(("127.0.0.1", 0))
        port = tmp.getsockname()[1]
    stop_event = threading.Event()
    srv_thread = threading.Thread(
        target=_loopback_server, args=(port, stop_event, buf_size), daemon=True
    )
    srv_thread.start()
    time.sleep(0.05)
    total = 0
    t0 = time.perf_counter()
    try:
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        cli.connect(("127.0.0.1", port))
        deadline = t0 + duration
        while time.perf_counter() < deadline:
            total += cli.send(payload)
        cli.close()
    except OSError:
        return 0.0
    finally:
        stop_event.set()
    elapsed = max(time.perf_counter() - t0, 1e-3)
    return (total * 8) / elapsed / 1e6


def bench_latency(n_pings: int = 8) -> float | None:
    samples: list[float] = []
    per_host = max(1, n_pings // len(_LATENCY_HOSTS))
    for host, port in _LATENCY_HOSTS:
        for _ in range(per_host):
            try:
                t0 = time.perf_counter()
                sock = socket.create_connection((host, port), timeout=1.5)
                samples.append((time.perf_counter() - t0) * 1000)
                sock.close()
            except OSError:
                continue
        if samples:
            break
    if not samples:
        return None
    samples.sort()
    return float(samples[len(samples) // 2])


def main() -> dict:
    emit(
        {
            "type": "benchmark_group_start",
            "name": "network",
            "display": "Network",
            "message": "Measuring download, loopback, and latency…",
        }
    )
    emit({"type": "progress", "test": "network", "pct": 5, "message": "Starting download probe…"})
    dl_mbps, dl_url = bench_download(_DL_DUR)
    emit(
        {
            "type": "progress",
            "test": "network",
            "pct": 45,
            "message": f"Download: {dl_mbps:.1f} Mbps — loopback…",
            "download_mbps": round(dl_mbps, 2),
        }
    )
    lb_mbps = bench_loopback(_LB_DUR)
    emit(
        {
            "type": "progress",
            "test": "network",
            "pct": 75,
            "message": f"Loopback: {lb_mbps:.0f} Mbps — latency…",
            "loopback_mbps": round(lb_mbps, 1),
        }
    )
    lat_ms = bench_latency()
    emit(
        {
            "type": "progress",
            "test": "network",
            "pct": 95,
            "message": f"Latency: {f'{lat_ms:.1f} ms' if lat_ms else 'n/a'} — scoring…",
        }
    )
    primary = dl_mbps if dl_mbps > 0.1 else max(lb_mbps * 0.05, 0.0)
    score = log_score(primary, _FLOOR, _CEIL)
    result = {
        "type": "result",
        "test": "network",
        "download_mbps": round(dl_mbps, 2),
        "loopback_mbps": round(lb_mbps, 1),
        "latency_ms": round(lat_ms, 2) if lat_ms else None,
        "download_source": dl_url,
        "score": round(score, 1),
        "thermal_throttled": False,
    }
    emit(result)
    return result


if __name__ == "__main__":
    main()
