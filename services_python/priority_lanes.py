"""Scheduler priority lanes (P0–P3) for DistribAI job queues.

Lower lane rank is served first: ``P0`` (critical) before ``P1`` (default),
then ``P2``, then ``P3`` (best-effort).
"""

from __future__ import annotations

from typing import Final

from services_python.schemas import PriorityTier

PRIORITY_LANES: Final[tuple[str, ...]] = ("P0", "P1", "P2", "P3")

# SQL CASE fragment: lower integer = higher scheduling preference.
PRIORITY_TIER_ORDER_SQL: Final[str] = """
CASE COALESCE(priority_tier, 'P1')
    WHEN 'P0' THEN 0
    WHEN 'P1' THEN 1
    WHEN 'P2' THEN 2
    WHEN 'P3' THEN 3
    ELSE 4
END
""".strip()

# Qualified for JOIN queries that alias jobs as ``j``.
PRIORITY_TIER_ORDER_SQL_J: Final[str] = """
CASE COALESCE(j.priority_tier, 'P1')
    WHEN 'P0' THEN 0
    WHEN 'P1' THEN 1
    WHEN 'P2' THEN 2
    WHEN 'P3' THEN 3
    ELSE 4
END
""".strip()


def normalize_priority_tier(value: str | None, default: str = "P1") -> str:
    """Return a canonical ``P0``–``P3`` tier string."""
    raw = str(value or default).strip().upper()
    if raw in PRIORITY_LANES:
        return raw
    # Accept bare digits or lowercase from older clients.
    aliases = {"0": "P0", "1": "P1", "2": "P2", "3": "P3"}
    if raw in aliases:
        return aliases[raw]
    return default


def priority_lane_rank(tier: str | None) -> int:
    """Numeric rank for sorting (0 = highest priority)."""
    normalized = normalize_priority_tier(tier)
    try:
        return PRIORITY_LANES.index(normalized)
    except ValueError:
        return len(PRIORITY_LANES)


def parse_priority_tier_filter(raw: str | None) -> set[str] | None:
    """Parse ``?priority_tier=P0,P1`` into a set of tiers, or None if unset."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"all", "*"}:
        return None
    wanted: set[str] = set()
    for part in text.split(","):
        tier = normalize_priority_tier(part.strip(), default="")
        if tier in PRIORITY_LANES:
            wanted.add(tier)
        elif part.strip():
            # Unknown token — keep as uppercase so filter matches nothing useful
            # rather than silently ignoring typos.
            wanted.add(part.strip().upper())
    return wanted or None


def is_valid_priority_tier(value: str | None) -> bool:
    """True when value is a known PriorityTier member."""
    raw = str(value or "").strip().upper()
    if raw in PRIORITY_LANES or raw in {"0", "1", "2", "3"}:
        return True
    try:
        PriorityTier(raw)
        return True
    except ValueError:
        return False
