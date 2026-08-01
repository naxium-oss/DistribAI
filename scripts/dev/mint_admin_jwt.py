#!/usr/bin/env python3
"""Mint an admin JWT for DistribAI orchestrator administration.

Run on the orchestrator host (or anywhere with access to JWT_SECRET) to
produce a Bearer token usable against /admin/* endpoints.

Usage:
    python scripts/dev/mint_admin_jwt.py                       # default: 24h TTL
    python scripts/dev/mint_admin_jwt.py --subject ops@team    # custom subject
    python scripts/dev/mint_admin_jwt.py --ttl-hours 720       # 30 days
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is importable when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jwt  # noqa: E402

from services_python.constants import JWT_ALGORITHM, JWT_SECRET  # noqa: E402


def mint(subject: str, ttl_hours: int) -> str:
    now = int(time.time())
    claims = {
        "sub": subject,
        "kind": "admin",
        "iat": now,
        "exp": now + ttl_hours * 3600,
    }
    return jwt.encode(claims, JWT_SECRET, algorithm=JWT_ALGORITHM)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="operator", help="Token subject (default: operator)")
    parser.add_argument(
        "--ttl-hours", type=int, default=24, help="Token TTL in hours (default: 24)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the token (suitable for shell capture)",
    )
    args = parser.parse_args()

    token = mint(args.subject, args.ttl_hours)
    if args.quiet:
        print(token)
    else:
        print(f"# DistribAI admin JWT (kind=admin, sub={args.subject}, ttl={args.ttl_hours}h)")
        print(
            f"# Usage: curl -H 'Authorization: Bearer {token[:24]}...' http://127.0.0.1:8766/admin/jobs"
        )
        print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
