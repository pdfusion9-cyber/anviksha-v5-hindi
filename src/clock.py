"""Bind a training run to the CPU's actual clock instead of a fixed step count.

A fixed "10,000 steps" means very different things on different machines: on a
3.5 GHz desktop core that is minutes, on a throttled phone core it is hours. So
the run is bounded by *cycles* rather than by steps.

Two numbers are measured, not assumed:

  1. **Clock speed** — read from /proc/cpuinfo, falling back to a wall-clock
     calibration if the field is missing (many containers hide it).
  2. **Steps per second** — measured by timing the first N steps of the actual
     model, not extrapolated from FLOPs.

The step target is then:

    steps = min(target_steps, steps_per_second * budget_seconds)

so a slow machine stops at the budget with fewer steps, and a fast one reaches
the target and stops early. Either way the run is honest about what it did.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path


def read_clock_ghz() -> float | None:
    """Current core clock in GHz, or None when the kernel does not expose it."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    text = cpuinfo.read_text(errors="ignore")
    values = [float(m) for m in re.findall(r"cpu MHz\s*:\s*([0-9.]+)", text)]
    if not values:
        return None
    return max(values) / 1000.0


def calibrate_clock_ghz(seconds: float = 0.3) -> float:
    """Estimate effective clock from a tight integer loop.

    This measures *effective* speed (including throttling and contention), which
    is what actually matters for scheduling, not the nominal nameplate frequency.
    """
    iterations = 0
    started = time.perf_counter()
    x = 0
    while time.perf_counter() - started < seconds:
        for _ in range(100_000):
            x += 1
        iterations += 100_000
    elapsed = time.perf_counter() - started
    # a tight add loop runs roughly one iteration per cycle on modern cores
    return (iterations / elapsed) / 1e9


@dataclass
class ClockBudget:
    clock_ghz: float
    clock_source: str
    budget_seconds: float
    cycles: float
    steps_per_second: float | None = None
    planned_steps: int | None = None

    def plan(self, target_steps: int, measured_steps_per_second: float) -> int:
        """Cap the step target so the run fits the cycle budget."""
        self.steps_per_second = measured_steps_per_second
        affordable = int(measured_steps_per_second * self.budget_seconds)
        self.planned_steps = min(target_steps, affordable)
        return self.planned_steps

    def as_dict(self) -> dict:
        return {
            "clock_ghz": round(self.clock_ghz, 3),
            "clock_source": self.clock_source,
            "budget_seconds": round(self.budget_seconds, 1),
            "budget_giga_cycles": round(self.cycles / 1e9, 1),
            "measured_steps_per_second": (
                round(self.steps_per_second, 3) if self.steps_per_second else None
            ),
            "planned_steps": self.planned_steps,
        }


def make_budget(budget_seconds: float) -> ClockBudget:
    clock = read_clock_ghz()
    if clock:
        source = "cpuinfo"
    else:
        clock = calibrate_clock_ghz()
        source = "calibrated"
    return ClockBudget(
        clock_ghz=clock,
        clock_source=source,
        budget_seconds=budget_seconds,
        cycles=clock * 1e9 * budget_seconds,
    )
