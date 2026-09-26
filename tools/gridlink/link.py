"""The shareable join link.

A join link is the only thing an operator has to send a contributor. It is
either a ``distribai://join`` URI or something a human would paste anyway (a
bare ``host:port`` or an ``https://…`` URL), which :meth:`GridLink.parse`
normalises.

::

    distribai://join?v=1&engine=native&provider=cloudflare
        &host=random-words.trycloudflare.com&port=443&tls=1
        &public=https%3A%2F%2Frandom-words.trycloudflare.com
        &name=my-grid&invite=team-alpha

The ``engine`` field says which stack answers the link. ``native`` is the C++
coordinator, which speaks HTTP on ``DEFAULT_GRID_PORT`` and serves the worker
API and the dashboard from one port. ``legacy`` is the frozen Python gRPC
control plane on ``DEFAULT_GRPC_PORT``.

Parsing is deliberately forgiving (workers paste whatever they were given) and
never raises on a bare host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit

DEFAULT_GRID_PORT = 50061  # native coordinator: worker API plus dashboard
DEFAULT_GRPC_PORT = 50051  # frozen legacy gRPC control plane
DEFAULT_ADMIN_PORT = 8766
LINK_SCHEME = "distribai"
LINK_VERSION = 1
ENGINES = ("native", "legacy")


def default_port(engine: str) -> int:
    return DEFAULT_GRPC_PORT if engine == "legacy" else DEFAULT_GRID_PORT


@dataclass
class GridLink:
    """Everything a worker needs to dial the grid and register."""

    host: str
    port: int = DEFAULT_GRID_PORT
    tls: bool = False
    provider: str = "local"
    engine: str = "native"
    public_url: str | None = None
    admin_url: str | None = None
    invite: str | None = None
    name: str | None = None

    # ---------------------------------------------------------------- url ----
    @property
    def orchestrator_url(self) -> str:
        """What a worker dials: an HTTP base URL for the native grid, a
        ``host:port`` gRPC target for the legacy stack."""
        if self.engine == "legacy":
            return f"{self.host}:{self.port}"
        return f"{'https' if self.tls else 'http'}://{self.host}:{self.port}"

    @property
    def is_native(self) -> bool:
        return self.engine == "native"

    @property
    def is_public(self) -> bool:
        return self.provider in {"cloudflare", "ngrok"}

    def to_uri(self) -> str:
        """Encode as a ``distribai://join`` URI."""
        params: list[tuple[str, str]] = [
            ("v", str(LINK_VERSION)),
            ("engine", self.engine),
            ("provider", self.provider),
            ("host", self.host),
            ("port", str(self.port)),
            ("tls", "1" if self.tls else "0"),
        ]
        if self.public_url:
            params.append(("public", self.public_url))
        if self.admin_url:
            params.append(("admin", self.admin_url))
        if self.invite:
            params.append(("invite", self.invite))
        if self.name:
            params.append(("name", self.name))
        query = "&".join(f"{quote(str(k))}={quote(str(v))}" for k, v in params)
        return urlunsplit((LINK_SCHEME, "join", "", query, ""))

    # -------------------------------------------------------------- parse ----
    @classmethod
    def parse(cls, text: str) -> GridLink:
        """Parse a join link, ``host:port`` host string, or http(s) URL."""
        raw = (text or "").strip().strip("'\"")
        if not raw:
            raise ValueError("empty join link")

        if raw.startswith(f"{LINK_SCHEME}://"):
            return cls._parse_uri(raw)

        split = urlsplit(raw)
        if split.scheme in {"http", "https", "grpc", "grpcs"}:
            if not split.hostname:
                raise ValueError(f"join link has no host: {text!r}")
            engine = "legacy" if split.scheme in {"grpc", "grpcs"} else "native"
            tls = split.scheme in {"https", "grpcs"}
            port = split.port or (443 if tls else default_port(engine))
            return cls(
                host=split.hostname,
                port=port,
                tls=tls,
                provider="tunnel" if tls else "local",
                engine=engine,
                public_url=urlunsplit((split.scheme, split.netloc, split.path, "", "")),
            )

        # Bare host / host:port. A bare target is a local or LAN orchestrator, and
        # with no scheme the native grid is the one to try first.
        host, _, port_text = raw.partition(":")
        host = host.strip()
        if not host:
            raise ValueError(f"join link has no host: {text!r}")
        port = int(port_text) if port_text.strip() else DEFAULT_GRID_PORT
        return cls(host=host, port=port, tls=False, provider="local")

    @classmethod
    def _parse_uri(cls, raw: str) -> GridLink:
        split = urlsplit(raw)
        q = parse_qs(split.query, keep_blank_values=True)

        def one(key: str) -> str | None:
            values = q.get(key)
            return values[0] if values else None

        # The netloc is the literal "join" pseudo-host, so the real host must
        # come from the query parameters.
        host = (one("host") or "").strip()
        if not host:
            raise ValueError("join link is missing 'host'")
        port_raw = one("port")
        provider = one("provider") or "local"
        tls_raw = one("tls")
        engine = (one("engine") or "native").strip().lower()
        if engine not in ENGINES:
            raise ValueError(f"unknown engine {engine!r} in join link")
        return cls(
            host=host,
            port=int(port_raw) if port_raw else default_port(engine),
            tls=(tls_raw == "1") if tls_raw is not None else provider in {"cloudflare", "ngrok"},
            provider=provider,
            engine=engine,
            public_url=one("public"),
            admin_url=one("admin"),
            invite=one("invite"),
            name=one("name"),
        )

    # ------------------------------------------------------------ display ----
    def summary(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "provider": self.provider,
            "host": self.host,
            "port": self.port,
            "tls": self.tls,
            "orchestrator_url": self.orchestrator_url,
            "public_url": self.public_url,
            "name": self.name,
        }

    def worker_command(self, *, binary: str = "build/cpp_port/distribai_worker") -> str:
        """The native worker command for this link."""
        cmd = f"{binary} --orchestrator {self.orchestrator_url}"
        if self.invite:
            cmd += f" --invite {self.invite}"
        return cmd

    def share_block(self, *, python: str = "python", binary: str | None = None) -> str:
        """A paste-anywhere block a contributor can follow verbatim."""
        title = f"Join the DistribAI grid{(' (' + self.name + ')') if self.name else ''}"
        lines = [
            "=" * 66,
            title,
            "=" * 66,
            "",
            "Join link (paste it into the join command below):",
            f"    {self.to_uri()}",
        ]
        if self.is_native:
            lines += [
                "",
                "Run the C++ worker on any machine. Build it once with",
                "    make grid",
                "then start it:",
                f"    {self.worker_command(binary=binary or 'build/cpp_port/distribai_worker')}",
            ]
            if self.admin_url:
                lines.append(f"    # dashboard: {self.admin_url}")
        else:
            lines += [
                "",
                "Or use the fields directly (frozen Python worker):",
                f"    ORCHESTRATOR_URL={self.orchestrator_url}",
                f"    GRPC_USE_TLS={'true' if self.tls else 'false'}",
            ]
            if self.invite:
                lines.append(f"    DISTRIBAI_INVITE_CODE={self.invite}")
            if self.admin_url:
                lines.append(f"    ADMIN_URL={self.admin_url}")
        lines += [
            "",
            "Either way, gridlink can print or run the plan for you:",
            f"    {python} -m gridlink join {self.to_uri()}",
            "",
            "Works on the free tiers: no paid plan and no custom domain needed.",
            "=" * 66,
        ]
        return "\n".join(lines)


def parse(text: str) -> GridLink:
    return GridLink.parse(text)
