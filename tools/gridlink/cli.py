"""gridlink CLI.

gridlink providers
gridlink expose --provider cloudflare [--port 50061] [--name my-grid]
gridlink join "distribai://join?..." [--ephemeral] [--exec]
gridlink link --host 10.0.0.5 --port 50061 --engine native

The native engine is the C++ coordinator (HTTP worker API plus dashboard). The
legacy engine is the frozen Python gRPC control plane.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from .expose import ExposeSession
from .join import WORKER_BINARY
from .join import join as do_join
from .link import ENGINES, GridLink, default_port
from .providers import TunnelError, available_providers


def _log(message: str) -> None:
    """Provider/tunnel chatter goes to stderr so stdout stays parseable."""
    print(message, file=sys.stderr)


def _add_engine(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--engine",
        default="native",
        choices=list(ENGINES),
        help="native = the C++ coordinator (default); legacy = the frozen Python gRPC stack",
    )


def _add_expose(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--provider",
        default="local",
        choices=["local", "cloudflare", "ngrok"],
        help="local = nothing exposed (default); cloudflare = free quick tunnel (no account); "
        "ngrok = free tier (needs NGROK_AUTHTOKEN)",
    )
    _add_engine(p)
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="port the coordinator listens on (default 50061 native, 50051 legacy)",
    )
    p.add_argument("--bind-host", default="127.0.0.1", help="local origin host (keep loopback)")
    p.add_argument("--link-host", default=None, help="host shown in the link for local provider")
    p.add_argument("--invite", default=None, help="invite code embedded in the link")
    p.add_argument("--name", default=None, help="grid name shown to contributors")
    p.add_argument("--download", action="store_true", help="download cloudflared if missing")
    p.add_argument(
        "--no-http2-origin",
        action="store_true",
        help="do not pass --http2-origin to cloudflared (gRPC needs it)",
    )
    p.add_argument("--timeout", type=float, default=45.0)
    p.add_argument("--once", action="store_true", help="print the link and exit (tunnel dies)")
    p.add_argument("--json", action="store_true")


def _add_join(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", nargs="?", default=None, help="join link, https URL, or host:port")
    p.add_argument("--orchestrator", default=None, help="host:port (alternative to the link)")
    p.add_argument("--node-id", default=None)
    p.add_argument("--state-dir", default=None)
    p.add_argument("--ephemeral", action="store_true", help="Colab/Kaggle/Molab: no disk state")
    p.add_argument("--invite", default=None)
    p.add_argument("--admin-url", default=None)
    _add_engine(p)
    p.add_argument(
        "--worker-bin",
        default=WORKER_BINARY,
        help=f"native worker binary (default {WORKER_BINARY})",
    )
    p.add_argument("--exec", dest="execute", action="store_true", help="actually start the worker")
    p.add_argument("--json", action="store_true")


def _add_link(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", required=True)
    _add_engine(p)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--tls", action="store_true")
    p.add_argument("--invite", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--admin-url", default=None)
    p.add_argument("--json", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridlink",
        description="Expose the DistribAI grid and join it from anywhere, for free.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _add_expose(sub.add_parser("expose", help="publish the grid and print a join link"))
    _add_join(sub.add_parser("join", help="join a grid from this machine"))
    _add_link(sub.add_parser("link", help="build a join link without running a tunnel"))
    providers = sub.add_parser("providers", help="show which tunnel providers are ready")
    providers.add_argument("--json", action="store_true")
    return parser


def _cmd_providers(args: argparse.Namespace) -> int:
    rows = [
        {
            "name": s.name,
            "available": s.available,
            "detail": s.detail,
            "install_hint": s.install_hint,
        }
        for s in available_providers()
    ]
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
        return 0
    print("Tunnel providers:")
    for row in rows:
        mark = "ready" if row["available"] else "  -- "
        print(f"  {mark}  {row['name']:11s} {row['detail']}")
        if not row["available"] and row["install_hint"]:
            print(f"         hint: {row['install_hint']}")
    print("\nCloudflare quick tunnels need no account or domain. Local bind stays the default.")
    return 0


def _cmd_expose(args: argparse.Namespace) -> int:
    session = ExposeSession(
        provider=args.provider,
        bind_host=args.bind_host,
        port=args.port or default_port(args.engine),
        engine=args.engine,
        link_host=args.link_host,
        invite=args.invite,
        name=args.name,
        http2_origin=not args.no_http2_origin,
        auto_download=args.download,
        timeout=args.timeout,
    )
    try:
        result = session.start(log=_log)
    except TunnelError as exc:
        print(f"gridlink: {exc}", file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.link.share_block())
        if result.provider != "local":
            print(f"\nLocal origin: http://{args.bind_host}:{args.port}  (keep it loopback)")
            if not args.once:
                print("\nTunnel is live. Press Ctrl-C to close it.")
    if result.provider == "local" or args.once:
        result.stop()
        return 0
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nclosing tunnel…")
    finally:
        result.stop()
    return 0


def _cmd_join(args: argparse.Namespace) -> int:
    target = args.target or args.orchestrator
    if not target:
        print("gridlink: provide a join link or --orchestrator host:port", file=sys.stderr)
        return 2
    try:
        plan = do_join(
            target,
            execute=args.execute,
            node_id=args.node_id,
            state_dir=args.state_dir,
            ephemeral=args.ephemeral,
            invite=args.invite,
            admin_url=args.admin_url,
            engine=args.engine,
            worker_bin=args.worker_bin,
        )
    except ValueError as exc:
        print(f"gridlink: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(plan.render())
        if args.execute:
            print("\nworker launched; watch your node appear in the operator's /admin/nodes")
        else:
            print("\n(plan only; re-run with --exec to start the worker)")
    return 0


def _cmd_link(args: argparse.Namespace) -> int:
    link = GridLink(
        host=args.host,
        port=args.port or default_port(args.engine),
        tls=args.tls,
        provider="tunnel" if args.tls else "local",
        engine=args.engine,
        invite=args.invite,
        name=args.name,
        admin_url=args.admin_url,
    )
    if args.json:
        print(json.dumps({"link": link.to_uri(), **link.summary()}, indent=2))
    else:
        print(link.share_block())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "expose":
        return _cmd_expose(args)
    if args.command == "join":
        return _cmd_join(args)
    if args.command == "link":
        return _cmd_link(args)
    if args.command == "providers":
        return _cmd_providers(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
