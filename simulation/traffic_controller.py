"""
traffic_controller.py
=====================
Traffic-signal controllers.

Signal cycle (one cycle = 6 phases):
    NS green -> NS yellow -> all red -> EW green -> EW yellow -> all red

A controller is described by a *plan*: an array of shape (K, 2) holding
(g_NS, g_EW) green durations in seconds for each of K equal time blocks of the
simulated period.  At the start of each block the cycle restarts with that
block's green times.  K = 1 gives the classical fixed-time controller.

The controller never moves vehicles; it only answers the question
"which approaches may discharge a vehicle at second t?" (see `capacity`).
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

APPROACHES: Tuple[str, ...] = ("N", "S", "E", "W")

# Signal state codes (one per second of simulation)
NS_GREEN, NS_YELLOW, RED_AFTER_NS, EW_GREEN, EW_YELLOW, RED_AFTER_EW = range(6)
STATE_NAMES = {NS_GREEN: "NS green", NS_YELLOW: "NS yellow", RED_AFTER_NS: "All red",
               EW_GREEN: "EW green", EW_YELLOW: "EW yellow", RED_AFTER_EW: "All red"}


@dataclass(frozen=True)
class SignalTiming:
    """Fixed safety / physics constants (NOT optimized)."""
    yellow: int = 3        # yellow duration (s)
    all_red: int = 1       # clearance time when everything is red (s)
    startup_lost: int = 2  # seconds after green starts before the first vehicle leaves
    headway: int = 2       # one vehicle leaves every `headway` s  (saturation flow = 1/headway veh/s)

    def lost_per_cycle(self) -> int:
        return 2 * (self.yellow + self.all_red)


class TrafficController:
    """Plan-based controller (base class for all controllers)."""
    name = "Plan controller"

    def __init__(self, plan, timing: SignalTiming = SignalTiming()):
        plan = np.asarray(plan, dtype=float).reshape(-1, 2)
        if plan.size == 0 or not np.all(np.isfinite(plan)):
            raise ValueError("plan must contain finite green times")
        self.plan = np.maximum(1, np.rint(plan)).astype(int)   # whole seconds, >= 1
        self.timing = timing

    @classmethod
    def from_vector(cls, x, timing: SignalTiming = SignalTiming()):
        """Build a controller from the flat optimizer vector
        [gNS_1, gEW_1, gNS_2, gEW_2, ...]."""
        return cls(np.asarray(x, dtype=float).reshape(-1, 2), timing)

    @property
    def n_blocks(self) -> int:
        return len(self.plan)

    def cycle_lengths(self) -> np.ndarray:
        return self.plan.sum(axis=1) + self.timing.lost_per_cycle()

    def timeline(self, horizon: int):
        """Return (state, elapsed): for every second the signal state code and
        the number of seconds since that phase began."""
        state = np.zeros(horizon, dtype=np.int8)
        elapsed = np.zeros(horizon, dtype=np.int32)
        K = self.n_blocks
        edges = (np.arange(K + 1) * horizon) // K          # block boundaries
        y, r = self.timing.yellow, self.timing.all_red
        for k in range(K):
            g_ns, g_ew = self.plan[k]
            phases = [(NS_GREEN, g_ns), (NS_YELLOW, y), (RED_AFTER_NS, r),
                      (EW_GREEN, g_ew), (EW_YELLOW, y), (RED_AFTER_EW, r)]
            t, end = int(edges[k]), int(edges[k + 1])
            while t < end:                                   # repeat the cycle until the block ends
                for code, dur in phases:
                    if dur <= 0:
                        continue
                    stop = min(t + int(dur), end)
                    state[t:stop] = code
                    elapsed[t:stop] = np.arange(stop - t)
                    t = stop
                    if t >= end:
                        break
        return state, elapsed

    def capacity(self, horizon: int):
        """Return (cap, state).  cap[t, j] = 1 if approach j (N,S,E,W) may
        release ONE vehicle at second t, else 0.  A vehicle can leave only
        during green, after the start-up lost time, once every `headway` s."""
        state, elapsed = self.timeline(horizon)
        lost, hw = self.timing.startup_lost, self.timing.headway
        slot = (elapsed >= lost) & (((elapsed - lost) % hw) == 0)
        cap_ns = ((state == NS_GREEN) & slot).astype(np.int64)
        cap_ew = ((state == EW_GREEN) & slot).astype(np.int64)
        return np.stack([cap_ns, cap_ns, cap_ew, cap_ew], axis=1), state


class FixedTimeController(TrafficController):
    """Conventional baseline: the same green times all day."""
    name = "Fixed-Time"

    def __init__(self, g_ns: int = 30, g_ew: int = 30, timing: SignalTiming = SignalTiming()):
        super().__init__([[g_ns, g_ew]], timing)
