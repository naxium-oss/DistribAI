"""
Credit Multiplier System (Production Implementation)
Implements credit earning multipliers as specified in README §5.1:
- 1.5x — Community Compute Opt-In (available for surge capacity)
- 1.2x — Reliability bonus (>99% uptime, trailing 7 days)
- 1.1x — Early adopter bonus (first 6 months post-launch)
- 0.8x — Penalty for elevated error rates (>2% task failures)
- 2.0x — Surge boost (admin-triggered network-wide bonus)
Also implements:
- Velocity caps (daily earning maximum)
- Demand-responsive issuance (0.8x when network utilization < 20%)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class NodeMultiplierState:
    node_id: str
    surge_opt_in: bool = False
    early_adopter: bool = True
    early_adopter_start: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0
    uptime_seconds: float = 0.0
    first_seen: float = field(default_factory=time.time)
    daily_earnings: float = 0.0
    daily_window_start: float = field(default_factory=time.time)

    def reliability_score(self) -> float:
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks == 0:
            return 1.0
        success_rate = self.tasks_completed / total_tasks
        uptime_hours = self.uptime_seconds / 3600
        uptime_bonus = min(0.1, uptime_hours / 168)
        return success_rate + uptime_bonus

    def is_reliable(self, threshold: float = 0.99) -> bool:
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks < 10:
            return False
        success_rate = self.tasks_completed / total_tasks
        return success_rate >= threshold

    def has_high_error_rate(self, threshold: float = 0.02) -> bool:
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks < 10:
            return False
        failure_rate = self.tasks_failed / total_tasks
        return failure_rate > threshold

    def early_adopter_active(self, duration_seconds: float = 15552000) -> bool:
        elapsed = time.time() - self.early_adopter_start
        return elapsed < duration_seconds


class CreditMultiplierEngine:
    """
    Engine for calculating credit earning multipliers.
    Implements the formula from README §5.1:
    credits_per_hour = compute_score × base_rate × multipliers
    """

    SURGE_OPT_IN_MULTIPLIER = 1.5
    RELIABILITY_BONUS_MULTIPLIER = 1.2
    EARLY_ADOPTER_MULTIPLIER = 1.1
    ERROR_PENALTY_MULTIPLIER = 0.8
    LOW_DEMAND_MULTIPLIER = 0.8
    SURGE_BOOST_MULTIPLIER = 2.0
    DEFAULT_DAILY_VELOCITY_CAP = 1000.0

    def __init__(
        self,
        velocity_cap: float = DEFAULT_DAILY_VELOCITY_CAP,
        launch_timestamp: float | None = None,
    ):
        self.velocity_cap = velocity_cap
        self.launch_timestamp = launch_timestamp or time.time()
        self.node_states: dict[str, NodeMultiplierState] = {}
        self.network_utilization: float = 0.5
        self.surge_active: bool = False
        self.surge_expires_at: float = 0.0

    def get_or_create_state(self, node_id: str) -> NodeMultiplierState:
        if node_id not in self.node_states:
            self.node_states[node_id] = NodeMultiplierState(
                node_id=node_id, early_adopter_start=self.launch_timestamp
            )
        return self.node_states[node_id]

    def calculate_multipliers(
        self, node_id: str, compute_score: float, surge_opt_in: bool | None = None
    ) -> dict:
        """
        Calculate all applicable multipliers for a node.
        Args:
            node_id: The node identifier
            compute_score: The node's compute benchmark score
            surge_opt_in: Whether node opts into surge capacity
        Returns:
            Dictionary with multiplier breakdown
        """
        state = self.get_or_create_state(node_id)
        if surge_opt_in is not None:
            state.surge_opt_in = surge_opt_in
        multipliers: dict[str, float | list[str] | None] = {
            "base": 1.0,
            "surge_opt_in": None,
            "reliability": None,
            "early_adopter": None,
            "error_penalty": None,
            "low_demand": None,
            "surge_boost": None,
            "final": 1.0,
        }
        applied: list[str] = []
        final_multiplier = 1.0
        if state.surge_opt_in:
            multipliers["surge_opt_in"] = self.SURGE_OPT_IN_MULTIPLIER
            final_multiplier *= self.SURGE_OPT_IN_MULTIPLIER
            applied.append(f"surge_opt_in:{self.SURGE_OPT_IN_MULTIPLIER}")
        if state.is_reliable(threshold=0.99):
            multipliers["reliability"] = self.RELIABILITY_BONUS_MULTIPLIER
            final_multiplier *= self.RELIABILITY_BONUS_MULTIPLIER
            applied.append(f"reliability:{self.RELIABILITY_BONUS_MULTIPLIER}")
        if state.early_adopter_active():
            multipliers["early_adopter"] = self.EARLY_ADOPTER_MULTIPLIER
            final_multiplier *= self.EARLY_ADOPTER_MULTIPLIER
            applied.append(f"early_adopter:{self.EARLY_ADOPTER_MULTIPLIER}")
        if state.has_high_error_rate(threshold=0.02):
            multipliers["error_penalty"] = self.ERROR_PENALTY_MULTIPLIER
            final_multiplier *= self.ERROR_PENALTY_MULTIPLIER
            applied.append(f"error_penalty:{self.ERROR_PENALTY_MULTIPLIER}")
        if self.network_utilization < 0.20:
            multipliers["low_demand"] = self.LOW_DEMAND_MULTIPLIER
            final_multiplier *= self.LOW_DEMAND_MULTIPLIER
            applied.append(f"low_demand:{self.LOW_DEMAND_MULTIPLIER}")
        if self.surge_active and time.time() < self.surge_expires_at:
            multipliers["surge_boost"] = self.SURGE_BOOST_MULTIPLIER
            final_multiplier *= self.SURGE_BOOST_MULTIPLIER
            applied.append(f"surge_boost:{self.SURGE_BOOST_MULTIPLIER}")
        multipliers["final"] = final_multiplier
        multipliers["applied"] = applied
        return multipliers

    def calculate_credits(
        self,
        node_id: str,
        compute_score: float,
        base_rate: float,
        hours: float = 1.0,
        surge_opt_in: bool | None = None,
    ) -> dict:
        """
        Calculate credits earned.
        Formula: credits = compute_score × base_rate × hours × multipliers
        Args:
            node_id: The node identifier
            compute_score: Benchmark score (e.g., TFLOPS equivalent)
            base_rate: Base earning rate per compute unit
            hours: Time period in hours
            surge_opt_in: Whether node opts into surge capacity
        Returns:
            Dictionary with credit calculation breakdown
        """
        multipliers = self.calculate_multipliers(node_id, compute_score, surge_opt_in)
        base_credits = compute_score * base_rate * hours
        adjusted_credits = base_credits * multipliers["final"]
        state = self.get_or_create_state(node_id)
        now = time.time()
        if now - state.daily_window_start > 86400:
            state.daily_earnings = 0.0
            state.daily_window_start = now
        remaining_cap = self.velocity_cap - state.daily_earnings
        if adjusted_credits > remaining_cap:
            capped_credits = remaining_cap
            velocity_capped = True
        else:
            capped_credits = adjusted_credits
            velocity_capped = False
        state.daily_earnings += capped_credits
        return {
            "node_id": node_id,
            "compute_score": compute_score,
            "base_rate": base_rate,
            "hours": hours,
            "base_credits": round(base_credits, 2),
            "multipliers": multipliers,
            "adjusted_credits": round(adjusted_credits, 2),
            "velocity_capped": velocity_capped,
            "capped_credits": round(capped_credits, 2),
            "daily_remaining": round(remaining_cap, 2),
            "daily_total": round(state.daily_earnings, 2),
        }

    def record_task_completion(self, node_id: str, success: bool, duration_seconds: float = 0):
        state = self.get_or_create_state(node_id)
        if success:
            state.tasks_completed += 1
        else:
            state.tasks_failed += 1
        state.uptime_seconds += duration_seconds

    def set_surge_opt_in(self, node_id: str, opt_in: bool):
        state = self.get_or_create_state(node_id)
        state.surge_opt_in = opt_in
        logger.info("Node %s... surge opt-in: %s", node_id[:20], opt_in)

    def trigger_surge(self, duration_seconds: float = 3600):
        """Activate network-wide surge boost for duration."""
        self.surge_active = True
        self.surge_expires_at = time.time() + duration_seconds
        logger.info("Surge boost activated for %s seconds", duration_seconds)

    def set_early_adopter(self, node_id: str, is_early_adopter: bool):
        state = self.get_or_create_state(node_id)
        state.early_adopter = is_early_adopter
        if is_early_adopter:
            state.early_adopter_start = time.time()
        logger.info("Node %s... early adopter status: %s", node_id[:20], is_early_adopter)

    def update_network_utilization(self, utilization: float):
        self.network_utilization = max(0.0, min(1.0, utilization))

    def get_stats(self) -> dict[str, Any]:
        """Aggregate multiplier engine stats for admin dashboards."""
        nodes = list(self.node_states.values())
        surge_opt_in = sum(1 for st in nodes if st.surge_opt_in)
        early_adopter = sum(1 for st in nodes if st.early_adopter_active())
        reliable = sum(1 for st in nodes if st.is_reliable())
        mult_sum = 0.0
        for st in nodes:
            mult_sum += float(self.calculate_multipliers(st.node_id, compute_score=0.0)["final"])
        avg_mult = mult_sum / len(nodes) if nodes else 1.0
        surge_remaining = max(0, int(self.surge_expires_at - time.time())) if self.surge_active else 0
        return {
            "tracked_nodes": len(nodes),
            "network_utilization": round(self.network_utilization, 6),
            "velocity_cap": self.velocity_cap,
            "surge_opt_in_nodes": surge_opt_in,
            "early_adopter_active_nodes": early_adopter,
            "reliable_nodes": reliable,
            "avg_effective_multiplier": round(avg_mult, 6),
            "surge_active": self.surge_active,
            "surge_remaining_seconds": surge_remaining,
        }

    def get_node_summary(self, node_id: str) -> dict:
        state = self.get_or_create_state(node_id)
        multipliers = self.calculate_multipliers(node_id, compute_score=0.0)
        return {
            "node_id": node_id,
            "surge_opt_in": state.surge_opt_in,
            "effective_multiplier": multipliers["final"],
            "multipliers": multipliers,
            "early_adopter_active": state.early_adopter_active(),
            "is_reliable": state.is_reliable(),
            "has_high_error_rate": state.has_high_error_rate(),
            "tasks_completed": state.tasks_completed,
            "tasks_failed": state.tasks_failed,
            "reliability_score": round(state.reliability_score(), 4),
            "uptime_hours": round(state.uptime_seconds / 3600, 2),
            "daily_earnings": round(state.daily_earnings, 2),
            "velocity_cap": self.velocity_cap,
        }


REFERENCE_EARNING_RATES = {
    "RTX 4090": {"base_credits_per_hour": 16, "compute_score": 82.6},
    "RTX 3090": {"base_credits_per_hour": 13, "compute_score": 71.0},
    "RTX 3080": {"base_credits_per_hour": 10, "compute_score": 54.0},
    "RTX 3060": {"base_credits_per_hour": 7, "compute_score": 38.0},
    "M2 Ultra": {"base_credits_per_hour": 8, "compute_score": 44.0},
    "M2 Pro": {"base_credits_per_hour": 5, "compute_score": 28.0},
}
