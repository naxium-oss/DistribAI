"""Tunnel providers: local (default), Cloudflare quick tunnel, ngrok.

Cloudflare Quick Tunnels need **no account and no domain**, and they are the free default.
ngrok works too when ``NGROK_AUTHTOKEN`` is set (free tier); it is optional.
Nothing is exposed unless a provider is asked for: ``local`` is the default and
runs no subprocess at all.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

TRYCLOUDFLARE_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")
NGROK_API = "http://127.0.0.1:4040/api/tunnels"
CLOUDFLARED_RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"
NGROK_AUTHTOKEN_ENV = "NGROK_AUTHTOKEN"


class TunnelError(RuntimeError):
    """A provider could not be made ready or its public URL never appeared."""


@dataclass
class ProviderStatus:
    name: str
    available: bool
    detail: str
    install_hint: str = ""


@dataclass
class TunnelHandle:
    provider: str
    url: str
    command: list[str]
    process: subprocess.Popen | None = None

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


# ---------------------------------------------------------------------------
# binary discovery / install
# ---------------------------------------------------------------------------


def bin_dir() -> Path:
    override = os.environ.get("GRIDLINK_BIN_DIR", "").strip()
    base = Path(override) if override else Path(__file__).resolve().parent / "bin"
    return base


def _cloudflared_asset() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(machine)
    if arch is None:
        raise TunnelError(f"unsupported CPU architecture for cloudflared: {machine}")
    if system == "linux":
        return f"cloudflared-linux-{arch}"
    if system == "darwin":
        return f"cloudflared-darwin-{arch}"
    if system == "windows":
        return f"cloudflared-windows-{arch}.exe"
    raise TunnelError(f"unsupported OS for cloudflared: {system}")


def find_cloudflared() -> Path | None:
    override = os.environ.get("CLOUDFLARED_BIN", "").strip()
    if override and Path(override).is_file():
        return Path(override)
    on_path = shutil.which("cloudflared")
    if on_path:
        return Path(on_path)
    local = bin_dir() / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
    return local if local.is_file() else None


def download_cloudflared(dest_dir: Path | None = None, *, log=print) -> Path:
    """Fetch the static cloudflared binary into the project-local bin dir."""
    dest_dir = dest_dir or bin_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    asset = _cloudflared_asset()
    url = f"{CLOUDFLARED_RELEASE}/{asset}"
    target = dest_dir / ("cloudflared.exe" if asset.endswith(".exe") else "cloudflared")
    log(f"downloading cloudflared: {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed GitHub host
            payload = resp.read()
    except OSError as exc:
        raise TunnelError(
            f"could not download cloudflared from {url}: {exc}\n"
            "Install it manually (https://developers.cloudflare.com/cloudflare-one/connections/"
            "connect-networks/downloads/) or set CLOUDFLARED_BIN."
        ) from exc
    target.write_bytes(payload)
    if not asset.endswith(".exe"):
        target.chmod(0o755)
    log(f"cloudflared installed at {target}")
    return target


def find_ngrok() -> Path | None:
    override = os.environ.get("NGROK_BIN", "").strip()
    if override and Path(override).is_file():
        return Path(override)
    on_path = shutil.which("ngrok")
    return Path(on_path) if on_path else None


# ---------------------------------------------------------------------------
# pure command / parsing helpers (unit-tested)
# ---------------------------------------------------------------------------


def cloudflared_command(
    binary: Path, bind_host: str, port: int, *, http2_origin: bool = True
) -> list[str]:
    """argv for a Cloudflare quick tunnel to ``http://bind_host:port``.

    ``--http2-origin`` makes cloudflared speak HTTP/2 to the origin, which is
    what a gRPC (h2c) server needs.
    """
    cmd = [str(binary), "tunnel", "--no-autoupdate", "--url", f"http://{bind_host}:{port}"]
    if http2_origin:
        cmd.append("--http2-origin")
    return cmd


def parse_cloudflared_url(text: str) -> str | None:
    """Extract the trycloudflare URL cloudflared prints on startup."""
    match = TRYCLOUDFLARE_RE.search(text or "")
    return match.group(0) if match else None


def ngrok_command(binary: Path, bind_host: str, port: int) -> list[str]:
    """argv for ``ngrok http``; the authtoken is passed via NGROK_AUTHTOKEN."""
    return [str(binary), "http", "--log=stdout", f"{bind_host}:{port}"]


def parse_ngrok_public_url(payload: str | dict) -> str | None:
    """Pick the https public_url from the ngrok local API response."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    tunnels = (payload or {}).get("tunnels") or []
    https = [
        t.get("public_url") for t in tunnels if str(t.get("public_url", "")).startswith("https://")
    ]
    if https:
        return https[0]
    for tunnel in tunnels:
        url = tunnel.get("public_url")
        if url:
            return str(url)
    return None


def _ngrok_env() -> dict[str, str]:
    env = dict(os.environ)
    token = os.environ.get(NGROK_AUTHTOKEN_ENV, "").strip()
    if token:
        env[NGROK_AUTHTOKEN_ENV] = token
    return env


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


class Provider:
    name: str = ""
    label: str = ""
    install_hint: str = ""

    def available(self) -> ProviderStatus:
        raise NotImplementedError

    def command(self, bind_host: str, port: int) -> list[str]:
        raise NotImplementedError

    def start(self, bind_host: str, port: int, *, timeout: float = 45.0, log=print) -> TunnelHandle:
        raise NotImplementedError


class LocalProvider(Provider):
    """No tunnel: report the local (or LAN) address as-is."""

    name = "local"
    label = "Local bind (default, nothing exposed)"

    def available(self) -> ProviderStatus:
        return ProviderStatus(self.name, True, "no tunnel, local address only")

    def command(self, bind_host: str, port: int) -> list[str]:
        return []

    def start(self, bind_host: str, port: int, *, timeout: float = 45.0, log=print) -> TunnelHandle:
        return TunnelHandle(provider=self.name, url=f"{bind_host}:{port}", command=[])


class CloudflareProvider(Provider):
    """Free Cloudflare Quick Tunnel: no account, no custom domain."""

    name = "cloudflare"
    label = "Cloudflare Quick Tunnel (free, no account)"

    def __init__(self, *, http2_origin: bool = True, auto_download: bool = True) -> None:
        self.http2_origin = http2_origin
        self.auto_download = auto_download

    def _binary(self, log=print) -> Path:
        found = find_cloudflared()
        if found is not None:
            return found
        if not self.auto_download:
            raise TunnelError(
                "cloudflared not found. Run `gridlink expose --provider cloudflare "
                "--download`, install it, or set CLOUDFLARED_BIN."
            )
        return download_cloudflared(log=log)

    def available(self) -> ProviderStatus:
        found = find_cloudflared()
        return ProviderStatus(
            self.name,
            True,
            f"cloudflared: {found}"
            if found
            else "cloudflared not installed (auto-download on first use)",
            "gridlink expose --provider cloudflare --download",
        )

    def command(self, bind_host: str, port: int) -> list[str]:
        binary = find_cloudflared()
        if binary is None:
            raise TunnelError("cloudflared not installed; call start() first")
        return cloudflared_command(binary, bind_host, port, http2_origin=self.http2_origin)

    def start(self, bind_host: str, port: int, *, timeout: float = 45.0, log=print) -> TunnelHandle:
        binary = self._binary(log=log)
        cmd = cloudflared_command(binary, bind_host, port, http2_origin=self.http2_origin)
        log("starting Cloudflare quick tunnel: " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        deadline = time.monotonic() + timeout
        url: str | None = None
        assert proc.stdout is not None
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            candidate = parse_cloudflared_url(line)
            if candidate and url is None:
                url = candidate
                log(f"public URL: {url}")
                break
        if url is None:
            log("cloudflared output did not contain a trycloudflare URL within the timeout")
            proc.terminate()
            raise TunnelError(
                "Cloudflare quick tunnel did not come up within "
                f"{int(timeout)}s. Check the cloudflared output / network access."
            )
        return TunnelHandle(provider=self.name, url=url, command=cmd, process=proc)


class NgrokProvider(Provider):
    """ngrok (free tier). Requires NGROK_AUTHTOKEN; optional, not the default."""

    name = "ngrok"
    label = "ngrok (free tier, needs NGROK_AUTHTOKEN)"

    def available(self) -> ProviderStatus:
        binary = find_ngrok()
        token = os.environ.get(NGROK_AUTHTOKEN_ENV, "").strip()
        problems = []
        if binary is None:
            problems.append("ngrok binary not on PATH (set NGROK_BIN)")
        if not token:
            problems.append(
                f"{NGROK_AUTHTOKEN_ENV} not set (free account: https://dashboard.ngrok.com)"
            )
        if problems:
            return ProviderStatus(self.name, False, "; ".join(problems), "gridlink providers")
        return ProviderStatus(self.name, True, f"ngrok: {binary}")

    def command(self, bind_host: str, port: int) -> list[str]:
        binary = find_ngrok()
        if binary is None:
            raise TunnelError("ngrok not installed; see https://ngrok.com/download")
        return ngrok_command(binary, bind_host, port)

    def start(self, bind_host: str, port: int, *, timeout: float = 45.0, log=print) -> TunnelHandle:
        status = self.available()
        if not status.available:
            raise TunnelError(
                "ngrok is not ready: "
                + status.detail
                + "\nThe ngrok path is optional. The free Cloudflare quick tunnel needs no account:"
                "\n    gridlink expose --provider cloudflare"
            )
        cmd = self.command(bind_host, port)
        log("starting ngrok tunnel: " + " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=_ngrok_env(),
        )
        deadline = time.monotonic() + timeout
        url: str | None = None
        while time.monotonic() < deadline:
            if proc.poll() is not None and url is None:
                raise TunnelError("ngrok exited before reporting a public URL")
            try:
                with urllib.request.urlopen(NGROK_API, timeout=3) as resp:  # noqa: S310 - loopback
                    url = parse_ngrok_public_url(resp.read().decode("utf-8", "replace"))
            except OSError:
                url = None
            if url:
                log(f"public URL: {url}")
                break
            time.sleep(1.0)
        if url is None:
            proc.terminate()
            raise TunnelError(
                f"ngrok did not report a public URL within {int(timeout)}s (is {NGROK_API} reachable?)"
            )
        return TunnelHandle(provider=self.name, url=url, command=cmd, process=proc)


PROVIDERS: dict[str, Provider] = {
    "local": LocalProvider(),
    "cloudflare": CloudflareProvider(),
    "ngrok": NgrokProvider(),
}


def get_provider(name: str) -> Provider:
    key = (name or "local").strip().lower()
    if key not in PROVIDERS:
        raise TunnelError(f"unknown provider {name!r}; choose one of {', '.join(PROVIDERS)}")
    return PROVIDERS[key]


def available_providers() -> list[ProviderStatus]:
    return [provider.available() for provider in PROVIDERS.values()]


__all__ = [
    "PROVIDERS",
    "CloudflareProvider",
    "LocalProvider",
    "NgrokProvider",
    "Provider",
    "ProviderStatus",
    "TunnelError",
    "TunnelHandle",
    "available_providers",
    "cloudflared_command",
    "download_cloudflared",
    "find_cloudflared",
    "find_ngrok",
    "get_provider",
    "ngrok_command",
    "parse_cloudflared_url",
    "parse_ngrok_public_url",
]


if __name__ == "__main__":  # tiny self-report: `python -m gridlink.providers`
    for status in available_providers():
        mark = "ok " if status.available else "-- "
        print(f"{mark}{status.name:11s} {status.detail}")
    sys.exit(0)
