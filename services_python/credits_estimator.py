"""Heuristic credit-cost estimates for job create UIs (non-billing authoritative)."""

from __future__ import annotations

from typing import Any

_TIER_BASE = {
    "P0": 1.5,
    "P1": 1.0,
    "P2": 0.75,
    "P3": 0.5,
}

_FAMILY_WEIGHT = {
    "decoder_transformer": 1.0,
    "moe_decoder": 1.35,
    "hybrid_attn_rnn": 1.15,
    "gru": 0.7,
    "lstm": 0.75,
    "gated_conv": 0.8,
    "resnet_lm": 0.85,
    "dense_ffn": 0.65,
}


def estimate_job_credits(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a transparent estimate from steps, tier, and architecture hints."""
    steps = max(1, int(payload.get("steps") or payload.get("total_steps") or 100))
    batch = max(1, int(payload.get("batch_size") or 32))
    tier = str(payload.get("priority_tier") or "P1").upper()
    tier_mult = _TIER_BASE.get(tier, 1.0)

    arch = payload.get("architecture_config") or {}
    if not isinstance(arch, dict):
        arch = {}
    family = str(arch.get("family") or arch.get("architecture") or "decoder_transformer").lower()
    family_mult = _FAMILY_WEIGHT.get(family, 1.0)
    dim = int(arch.get("dim") or 128)
    layers = int(arch.get("n_logical_layers") or arch.get("n_unique_layers") or arch.get("gru_layers") or 6)
    size_mult = max(0.5, (dim / 128.0) * (layers / 6.0) ** 0.5)

    opcodes = steps * batch
    credits = round(opcodes * 0.001 * tier_mult * family_mult * size_mult, 4)
    return {
        "estimate": True,
        "credits": credits,
        "opcodes": opcodes,
        "priority_tier": tier,
        "family": family,
        "factors": {
            "tier_multiplier": tier_mult,
            "family_multiplier": family_mult,
            "size_multiplier": round(size_mult, 4),
            "steps": steps,
            "batch_size": batch,
        },
        "disclaimer": "Estimate only — ledger settlement uses live multipliers and verified work.",
    }
