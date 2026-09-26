"""Expose the coordinator's control plane and produce a shareable join link.

Default is a plain **local bind**: no tunnel, no subprocess, nothing public.
Ask for ``--provider cloudflare`` (free quick tunnel, no account) or
``--provider ngrok`` (free tier, needs an authtoken) to publish it. The native
coordinator answers HTTP, so the tunnel proxies it without extra flags; the
legacy gRPC stack needs cloudflared's h2c origin flag, which the provider sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .link import DEFAULT_GRID_PORT, GridLink, default_port
from .providers import Provider, TunnelError, TunnelHandle, get_provider

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"}


@dataclass
class ExposeResult:
    link: GridLink
    provider: str
    public_url: str | None
    command: list[str] = field(default_factory=list)
    _handle: TunnelHandle | None = None

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.stop()

    def to_dict(self) -> dict:
        return {
            "engine": self.link.engine,
            "provider": self.provider,
            "public_url": self.public_url,
            "link": self.link.to_uri(),
            "orchestrator_url": self.link.orchestrator_url,
            "tls": self.link.tls,
            "command": list(self.command),
        }


def _link_from_public_url(provider: str, public_url: str, engine: str) -> tuple[str, int, bool]:
    """Split a public tunnel URL into (host, port, tls) for the connect target."""
    split = urlsplit(public_url)
    host = split.hostname
    if not host:
        raise TunnelError(f"tunnel reported an unusable public URL: {public_url!r}")
    tls = split.scheme in {"https", "grpcs"}
    port = split.port or (443 if tls else default_port(engine))
    return host, port, tls


class ExposeSession:
    """One expose run: bind config + provider + the resulting join link."""

    def __init__(
        self,
        *,
        provider: str = "local",
        bind_host: str = "127.0.0.1",
        port: int = DEFAULT_GRID_PORT,
        engine: str = "native",
        link_host: str | None = None,
        invite: str | None = None,
        name: str | None = None,
        http2_origin: bool = True,
        auto_download: bool = True,
        timeout: float = 45.0,
    ) -> None:
        self.provider_name = provider
        self.bind_host = bind_host
        self.port = port
        self.engine = engine
        self.link_host = link_host
        self.invite = invite
        self.name = name
        self.http2_origin = http2_origin
        self.auto_download = auto_download
        self.timeout = timeout
        self._provider: Provider = get_provider(provider)
        self._handle: TunnelHandle | None = None

    def local_link(self) -> GridLink:
        """The link for the no-tunnel case (also the fallback for tunnel-mode parse)."""
        host = self.link_host or self.bind_host
        return GridLink(
            host=host,
            port=self.port,
            tls=False,
            provider="local",
            engine=self.engine,
            invite=self.invite,
            name=self.name,
        )

    def start(self, *, log=print) -> ExposeResult:
        if self.provider_name == "local":
            link = self.local_link()
            log(f"local bind only: {link.orchestrator_url} (nothing exposed)")
            return ExposeResult(
                link=link, provider="local", public_url=None, command=[], _handle=None
            )

        # The tunnel dials the local origin from the same machine; the origin
        # stays on the loopback bind unless the operator changed it.
        handle = self._provider.start(self.bind_host, self.port, timeout=self.timeout, log=log)
        self._handle = handle
        host, port, tls = _link_from_public_url(self.provider_name, handle.url, self.engine)
        link = GridLink(
            host=host,
            port=port,
            tls=tls,
            provider=self.provider_name,
            engine=self.engine,
            public_url=handle.url,
            invite=self.invite,
            name=self.name,
        )
        return ExposeResult(
            link=link,
            provider=self.provider_name,
            public_url=handle.url,
            command=handle.command,
            _handle=handle,
        )

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.stop()
            self._handle = None


def expose(
    *,
    provider: str = "local",
    bind_host: str = "127.0.0.1",
    port: int = DEFAULT_GRID_PORT,
    engine: str = "native",
    link_host: str | None = None,
    invite: str | None = None,
    name: str | None = None,
    http2_origin: bool = True,
    auto_download: bool = True,
    timeout: float = 45.0,
    log=print,
) -> ExposeResult:
    """One-shot helper: build a session, start it, return the result."""
    session = ExposeSession(
        provider=provider,
        bind_host=bind_host,
        port=port,
        engine=engine,
        link_host=link_host,
        invite=invite,
        name=name,
        http2_origin=http2_origin,
        auto_download=auto_download,
        timeout=timeout,
    )
    return session.start(log=log)


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


__all__ = ["ExposeResult", "ExposeSession", "expose", "is_loopback"]
