"""Menstrual cycle phase resolver.

28-day average cycle assumed unless overridden. Phase boundaries match
the consensus model used by Stanford Lifestyle Medicine + ACOG patient
ed. The output drives nutrition + training adaptations elsewhere in
fitapp-core.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Literal, TypedDict

Phase = Literal["menstrual", "follicular", "ovulatory", "luteal"]


class CyclePhase(TypedDict):
    phase: Phase
    day: int
    cycle_length: int
    days_until_period: int
    next_period_iso: str


def cycle_phase(
    last_period_iso: str,
    today_iso: str | None = None,
    cycle_length: int = 28,
) -> CyclePhase:
    """Return cycle phase + day, given last period start (YYYY-MM-DD).

    Phases (default 28-day cycle):
        menstrual:   days 1–5
        follicular:  days 6–13
        ovulatory:   days 14–16
        luteal:      days 17–28+

    For non-28-day cycles the boundaries scale proportionally.
    """
    last = date.fromisoformat(last_period_iso)
    today = date.fromisoformat(today_iso) if today_iso else date.today()
    if cycle_length < 21 or cycle_length > 45:
        raise ValueError(f"cycle_length out of range: {cycle_length}")

    days_since = (today - last).days
    if days_since < 0:
        raise ValueError("last_period_iso must be on or before today_iso")

    cycle_day = (days_since % cycle_length) + 1
    days_until_period = cycle_length - cycle_day + 1
    next_period = today + timedelta(days=days_until_period - 1)

    # Scale boundaries proportionally; clamp to known anchors at 28-day base
    menstrual_end = max(3, round(cycle_length * 5 / 28))
    follicular_end = max(menstrual_end + 4, round(cycle_length * 13 / 28))
    ovulatory_end = max(follicular_end + 2, round(cycle_length * 16 / 28))

    if cycle_day <= menstrual_end:
        phase: Phase = "menstrual"
    elif cycle_day <= follicular_end:
        phase = "follicular"
    elif cycle_day <= ovulatory_end:
        phase = "ovulatory"
    else:
        phase = "luteal"

    return {
        "phase": phase,
        "day": cycle_day,
        "cycle_length": cycle_length,
        "days_until_period": days_until_period,
        "next_period_iso": next_period.isoformat(),
    }
