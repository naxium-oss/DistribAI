"""
Scheduler Configuration for Node Compute Donation

Manages timezone-aware contribution schedules and compute profiles.
Allows users to set:
- Specific times to contribute (start/end with timezone)
- Different compute levels per time slot
- Per-job compute boosts
- Compute percentage with GB display

All times stored internally as UTC, displayed in user's local timezone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from datetime import time as dt_time
from enum import Enum
from pathlib import Path


class ComputePriority(Enum):
    """Priority levels for job selection."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class ComputeProfile:
    """
    Compute donation profile.

    Controls how much of the node's resources are donated.

    Attributes:
        gpu_percent: Percentage of GPU to donate (0-100)
        vram_limit_gb: Maximum VRAM to use in GB (None = all)
        cpu_percent: Percentage of CPU to donate (0-100)
        priority: Job selection priority
        job_whitelist: Specific job IDs to prioritize (empty = all)
        job_blacklist: Job IDs to never accept

    Example:
        >>> profile = ComputeProfile(
        ...     gpu_percent=25,
        ...     vram_limit_gb=4.0,
        ...     priority=ComputePriority.HIGH
        ... )
    """

    gpu_percent: float = 90.0  # Default 90%
    vram_limit_gb: float | None = None
    cpu_percent: float = 50.0
    priority: ComputePriority = ComputePriority.NORMAL
    job_whitelist: list[str] = field(default_factory=list)
    job_blacklist: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate values."""
        self.gpu_percent = max(0.0, min(100.0, self.gpu_percent))
        self.cpu_percent = max(0.0, min(100.0, self.cpu_percent))

    @property
    def gpu_fraction(self) -> float:
        """GPU fraction as 0.0-1.0."""
        return self.gpu_percent / 100.0

    @property
    def cpu_fraction(self) -> float:
        """CPU fraction as 0.0-1.0."""
        return self.cpu_percent / 100.0

    def format_gpu_display(self, total_vram_gb: float) -> str:
        """
        Format GPU info for display: '25% (3GB of 12GB)'

        Args:
            total_vram_gb: Total GPU VRAM available

        Returns:
            Formatted string
        """
        if self.vram_limit_gb is not None:
            actual_gb = min(self.vram_limit_gb, total_vram_gb * self.gpu_fraction)
        else:
            actual_gb = total_vram_gb * self.gpu_fraction

        return f"{self.gpu_percent:.0f}% ({actual_gb:.1f}GB of {total_vram_gb:.1f}GB)"


@dataclass
class TimeSlot:
    """
    A time slot for contribution with timezone support.

    All times stored as UTC internally but can be set using any timezone.

    Attributes:
        start_time: Start time (UTC)
        end_time: End time (UTC)
        days_of_week: Which days this applies to (0=Monday, 6=Sunday)
        compute_profile: Profile to use during this slot
        enabled: Whether this slot is active

    Example:
        >>> slot = TimeSlot.from_local(
        ...     start_time=dt_time(9, 0),
        ...     end_time=dt_time(17, 0),
        ...     timezone_str='America/New_York',
        ...     compute_profile=ComputeProfile(gpu_percent=25)
        ... )
    """

    start_time: dt_time = field(default_factory=lambda: dt_time(9, 0))
    end_time: dt_time = field(default_factory=lambda: dt_time(17, 0))
    days_of_week: list[int] = field(default_factory=lambda: list(range(7)))  # All days
    compute_profile: ComputeProfile = field(default_factory=ComputeProfile)
    enabled: bool = True
    label: str = "Default Schedule"

    @classmethod
    def from_local(
        cls,
        start_time: dt_time,
        end_time: dt_time,
        timezone_offset_hours: float,
        compute_profile: ComputeProfile | None = None,
        days_of_week: list[int] | None = None,
        label: str = "Custom Schedule",
    ) -> TimeSlot:
        """
        Create a TimeSlot from local time, converting to UTC.

        Args:
            start_time: Local start time
            end_time: Local end time
            timezone_offset_hours: Hours offset from UTC (e.g., -5 for EST)
            compute_profile: Compute profile for this slot
            days_of_week: Which days (0=Monday)
            label: Human-readable label

        Returns:
            TimeSlot with UTC times
        """
        # Convert to UTC by subtracting offset
        utc_offset = timedelta(hours=timezone_offset_hours)

        # Create datetime objects for conversion
        dummy_date = datetime(2024, 1, 1)

        start_dt = datetime.combine(dummy_date, start_time) - utc_offset
        end_dt = datetime.combine(dummy_date, end_time) - utc_offset

        # Handle wrap-around (e.g., 11 PM EST = 4 AM UTC next day)
        start_utc = start_dt.time()
        end_utc = end_dt.time()

        return cls(
            start_time=start_utc,
            end_time=end_utc,
            days_of_week=days_of_week or list(range(7)),
            compute_profile=compute_profile or ComputeProfile(),
            label=label,
        )

    def is_active(self, dt: datetime | None = None) -> bool:
        """
        Check if this time slot is currently active.

        Args:
            dt: Datetime to check (default: current UTC time)

        Returns:
            True if current time falls within this slot
        """
        if dt is None:
            dt = datetime.now(UTC)

        # Check day of week
        if dt.weekday() not in self.days_of_week:
            return False

        if not self.enabled:
            return False

        # Check time
        current_time = dt.time()

        # Handle overnight slots (e.g., 22:00 - 06:00)
        if self.start_time > self.end_time:
            # Slot crosses midnight
            return current_time >= self.start_time or current_time <= self.end_time
        else:
            # Normal slot
            return self.start_time <= current_time <= self.end_time

    def to_local_display(self, timezone_offset_hours: float) -> dict:
        """
        Convert to local time for display.

        Args:
            timezone_offset_hours: Local timezone offset from UTC

        Returns:
            Dict with local times and display strings
        """
        utc_offset = timedelta(hours=timezone_offset_hours)

        dummy_date = datetime(2024, 1, 1)

        start_dt = datetime.combine(dummy_date, self.start_time) + utc_offset
        end_dt = datetime.combine(dummy_date, self.end_time) + utc_offset

        return {
            "label": self.label,
            "start_time": start_dt.strftime("%I:%M %p").lstrip("0"),
            "end_time": end_dt.strftime("%I:%M %p").lstrip("0"),
            "timezone_offset": f"UTC{timezone_offset_hours:+.1f}",
            "days": [self._day_name(d) for d in self.days_of_week],
            "enabled": self.enabled,
            "compute_profile": {
                "gpu_percent": self.compute_profile.gpu_percent,
                "cpu_percent": self.compute_profile.cpu_percent,
            },
        }

    @staticmethod
    def _day_name(day_num: int) -> str:
        """Get day name from number."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[day_num]


@dataclass
class ScheduleManager:
    """
    Manages multiple time slots and computes the active profile.

    Handles overlapping schedules by selecting the highest GPU percentage.

    Attributes:
        time_slots: List of scheduled time slots
        default_profile: Profile used when no schedule is active
        timezone_offset_hours: Local timezone offset from UTC
        auto_join: Whether to auto-connect at schedule start
        auto_leave: Whether to auto-disconnect at schedule end

    Example:
        >>> manager = ScheduleManager(timezone_offset_hours=-5)
        >>> manager.add_slot(TimeSlot.from_local(...))
        >>> active_profile = manager.get_active_profile()
    """

    time_slots: list[TimeSlot] = field(default_factory=list)
    default_profile: ComputeProfile = field(default_factory=lambda: ComputeProfile(gpu_percent=0))
    timezone_offset_hours: float = 0.0
    auto_join: bool = False
    auto_leave: bool = True
    config_path: Path = field(default_factory=lambda: Path.home() / ".distribai" / "schedule.json")

    def __post_init__(self):
        """Ensure config directory exists."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def add_slot(self, slot: TimeSlot) -> None:
        """Add a time slot."""
        self.time_slots.append(slot)

    def remove_slot(self, index: int) -> None:
        """Remove a time slot by index."""
        if 0 <= index < len(self.time_slots):
            del self.time_slots[index]

    def get_active_profile(self, dt: datetime | None = None) -> ComputeProfile:
        """
        Get the compute profile that should be active now.

        If multiple slots are active, returns the one with highest GPU percentage.
        If no slots active, returns default_profile.

        Args:
            dt: Datetime to check (default: now)

        Returns:
            Active ComputeProfile
        """
        if dt is None:
            dt = datetime.now(UTC)

        active_slots = [s for s in self.time_slots if s.is_active(dt)]

        if not active_slots:
            return self.default_profile

        # Return slot with highest GPU percentage
        best_slot = max(active_slots, key=lambda s: s.compute_profile.gpu_percent)
        return best_slot.compute_profile

    def should_be_contributing(self, dt: datetime | None = None) -> bool:
        """
        Check if node should be contributing based on schedule.

        Args:
            dt: Datetime to check (default: now)

        Returns:
            True if any enabled slot is active
        """
        profile = self.get_active_profile(dt)
        return profile.gpu_percent > 0

    def get_time_until_change(self, dt: datetime | None = None) -> float | None:
        """
        Get seconds until the active profile changes.

        Args:
            dt: Datetime to check (default: now)

        Returns:
            Seconds until change, or None if no schedule
        """
        if dt is None:
            dt = datetime.now(UTC)

        if not self.time_slots:
            return None

        # Check every minute for the next 24 hours
        for minutes in range(1, 24 * 60):
            future = dt + timedelta(minutes=minutes)
            current_profile = self.get_active_profile(dt)
            future_profile = self.get_active_profile(future)

            if current_profile.gpu_percent != future_profile.gpu_percent:
                return minutes * 60

        return None

    def save(self) -> None:
        """Save schedule to disk."""
        data = {
            "timezone_offset_hours": self.timezone_offset_hours,
            "auto_join": self.auto_join,
            "auto_leave": self.auto_leave,
            "default_profile": {
                "gpu_percent": self.default_profile.gpu_percent,
                "vram_limit_gb": self.default_profile.vram_limit_gb,
                "cpu_percent": self.default_profile.cpu_percent,
                "priority": self.default_profile.priority.value,
            },
            "time_slots": [],
        }

        for slot in self.time_slots:
            slot_data = {
                "label": slot.label,
                "start_time": slot.start_time.isoformat(),
                "end_time": slot.end_time.isoformat(),
                "days_of_week": slot.days_of_week,
                "enabled": slot.enabled,
                "compute_profile": {
                    "gpu_percent": slot.compute_profile.gpu_percent,
                    "vram_limit_gb": slot.compute_profile.vram_limit_gb,
                    "cpu_percent": slot.compute_profile.cpu_percent,
                    "priority": slot.compute_profile.priority.value,
                },
            }
            data["time_slots"].append(slot_data)

        self.config_path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> ScheduleManager:
        """Load schedule from disk."""
        if path is None:
            path = Path.home() / ".distribai" / "schedule.json"

        if not path.exists():
            return cls(config_path=path)

        try:
            data = json.loads(path.read_text())

            # Load default profile
            default_data = data.get("default_profile", {})
            default_profile = ComputeProfile(
                gpu_percent=default_data.get("gpu_percent", 0),
                vram_limit_gb=default_data.get("vram_limit_gb"),
                cpu_percent=default_data.get("cpu_percent", 50),
                priority=ComputePriority(default_data.get("priority", 2)),
            )

            # Load time slots
            slots = []
            for slot_data in data.get("time_slots", []):
                profile_data = slot_data.get("compute_profile", {})
                profile = ComputeProfile(
                    gpu_percent=profile_data.get("gpu_percent", 90),
                    vram_limit_gb=profile_data.get("vram_limit_gb"),
                    cpu_percent=profile_data.get("cpu_percent", 50),
                    priority=ComputePriority(profile_data.get("priority", 2)),
                )

                slot = TimeSlot(
                    start_time=dt_time.fromisoformat(slot_data["start_time"]),
                    end_time=dt_time.fromisoformat(slot_data["end_time"]),
                    days_of_week=slot_data.get("days_of_week", list(range(7))),
                    compute_profile=profile,
                    enabled=slot_data.get("enabled", True),
                    label=slot_data.get("label", "Schedule"),
                )
                slots.append(slot)

            return cls(
                time_slots=slots,
                default_profile=default_profile,
                timezone_offset_hours=data.get("timezone_offset_hours", 0),
                auto_join=data.get("auto_join", False),
                auto_leave=data.get("auto_leave", True),
                config_path=path,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading schedule: {e}")
            return cls(config_path=path)

    def to_display_dict(self, total_vram_gb: float) -> dict:
        """
        Convert to dict for GUI display.

        Args:
            total_vram_gb: Total GPU VRAM for display calculations

        Returns:
            Dict with all schedule info for UI
        """
        now = datetime.now(UTC)
        active_profile = self.get_active_profile(now)

        return {
            "timezone_offset_hours": self.timezone_offset_hours,
            "auto_join": self.auto_join,
            "auto_leave": self.auto_leave,
            "should_contribute": self.should_be_contributing(now),
            "active_profile": {
                "gpu_percent": active_profile.gpu_percent,
                "cpu_percent": active_profile.cpu_percent,
                "vram_limit_gb": active_profile.vram_limit_gb,
                "display": active_profile.format_gpu_display(total_vram_gb),
            },
            "time_slots": [
                slot.to_local_display(self.timezone_offset_hours) for slot in self.time_slots
            ],
        }


def detect_local_timezone() -> float:
    """
    Detect local timezone offset from UTC.

    Returns:
        Offset in hours (e.g., -5.0 for EST)
    """
    # Get current timezone offset
    local_time = time.localtime()
    if local_time.tm_isdst:
        offset_seconds = -time.altzone
    else:
        offset_seconds = -time.timezone

    return offset_seconds / 3600


# Global schedule manager instance
_schedule_manager: ScheduleManager | None = None


def get_schedule_manager() -> ScheduleManager:
    """Get or create global schedule manager."""
    global _schedule_manager
    if _schedule_manager is None:
        _schedule_manager = ScheduleManager.load()
    return _schedule_manager
