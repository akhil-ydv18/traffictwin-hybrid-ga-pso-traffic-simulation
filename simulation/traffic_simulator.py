"""
traffic_simulator.py
====================
Discrete-time (1 s) simulator of a single four-way intersection.

Model (per approach j in N,S,E,W, per second t)
    q[t] = max(0, q[t-1] + a[t] - c[t])
where a[t] ~ Poisson(lambda_j(t)) are the arrivals and c[t] in {0,1} is the
signal capacity from TrafficController.capacity().

This is a "Lindley recursion".  It can be solved for the WHOLE time axis with
three NumPy calls (cumsum, minimum.accumulate, subtraction) instead of a slow
Python loop, which makes one simulation take well under a millisecond.  That
speed is what makes optimisation with thousands of evaluations practical.

Metric definitions
    avg_wait   : total vehicle-seconds spent in queues / number of vehicles that
                 arrived.  (= area under the queue curve; vehicles still queued
                 at the end are charged their waiting time so far.)
    avg_queue  : mean queue length per approach, averaged over time.
    max_queue  : largest queue seen on any approach at any time.
    avg_travel : free-flow travel time + avg_wait.
    served     : vehicles that crossed the stop line.
    throughput : served vehicles per hour.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .intersection import run_object_engine
from .traffic_controller import APPROACHES, SignalTiming, TrafficController
from .vehicle import Vehicle, vehicles_from_counts

# ----------------------------------------------------------------------------
# Scenarios.  rates = vehicles PER MINUTE on approach (N, S, E, W).
# `shift` makes demand change during the run: in the first third the NS axis is
# (1+shift) times busier and EW (1-shift) times; the middle third is balanced;
# in the last third the roles swap.  This is why time-dependent signal plans
# (K > 1) can beat a single fixed plan.
# ----------------------------------------------------------------------------
SCENARIOS: Dict[str, Dict] = {
    "Low":    dict(rates=(3.0, 3.0, 2.5, 2.5),     shift=0.3),
    "Medium": dict(rates=(6.0, 6.0, 5.0, 5.0),     shift=0.3),
    "High":   dict(rates=(8.5, 8.5, 7.0, 7.0),     shift=0.3),
    "Peak":   dict(rates=(10.5, 10.5, 9.0, 9.0),   shift=0.3),
}


@dataclass
class SimConfig:
    duration: int = 3600          # simulated seconds
    yellow: int = 3
    all_red: int = 1
    startup_lost: int = 2
    headway: int = 2
    free_flow_time: float = 20.0  # seconds to drive the approach road without any delay

    def __post_init__(self):
        if self.duration < 60:
            raise ValueError("duration must be at least 60 s")
        if self.headway < 1 or self.yellow < 0 or self.all_red < 0 or self.startup_lost < 0:
            raise ValueError("invalid signal timing constants")

    @property
    def timing(self) -> SignalTiming:
        return SignalTiming(self.yellow, self.all_red, self.startup_lost, self.headway)


def build_demand(rates_per_min: Sequence[float], shift: float, duration: int) -> np.ndarray:
    """Return a (duration, 4) array of Poisson means (vehicles per second)."""
    rates = np.asarray(rates_per_min, dtype=float)
    if rates.shape != (4,) or np.any(rates < 0):
        raise ValueError("rates must be four non-negative numbers (N, S, E, W)")
    if not 0 <= shift < 1:
        raise ValueError("shift must be in [0, 1)")
    third = (np.arange(duration) * 3) // duration                 # 0,1,2
    ns_factor = np.array([1 + shift, 1.0, 1 - shift])[third]
    ew_factor = np.array([1 - shift, 1.0, 1 + shift])[third]
    factors = np.stack([ns_factor, ns_factor, ew_factor, ew_factor], axis=1)
    return rates[None, :] / 60.0 * factors


def scenario_demand(name: str, duration: int) -> np.ndarray:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario '{name}', choose from {list(SCENARIOS)}")
    s = SCENARIOS[name]
    return build_demand(s["rates"], s["shift"], duration)


@dataclass
class SimResult:
    metrics: Dict[str, float]
    queue_ts: np.ndarray                      # (T, 4) queue length after each second
    dep_ts: np.ndarray                        # (T, 4) departures in each second
    arrivals: np.ndarray                      # (T, 4)
    state_ts: np.ndarray                      # (T,) signal state codes
    vehicles: Optional[List[Vehicle]] = field(default=None)


class TrafficSimulator:
    def __init__(self, config: SimConfig = SimConfig()):
        self.cfg = config

    # ------------------------------------------------------------------
    def generate_arrivals(self, demand: np.ndarray, seed: int) -> np.ndarray:
        """Random vehicle arrivals.  Same seed -> same traffic (common random numbers)."""
        rng = np.random.default_rng(seed)
        return rng.poisson(demand).astype(np.int64)

    # ------------------------------------------------------------------
    @staticmethod
    def _queues_fast(arrivals: np.ndarray, cap: np.ndarray):
        """Vectorised Lindley recursion q[t] = max(0, q[t-1] + a[t] - c[t])."""
        x = np.cumsum(arrivals - cap, axis=0)
        floor = np.minimum.accumulate(np.minimum(x, 0), axis=0)
        q = x - floor
        prev = np.vstack([np.zeros((1, q.shape[1]), dtype=q.dtype), q[:-1]])
        dep = prev + arrivals - q
        return q, dep

    def _metrics(self, q, dep, arrivals) -> Dict[str, float]:
        T = q.shape[0]
        n_arr = int(arrivals.sum())
        avg_wait = float(q.sum() / n_arr) if n_arr else 0.0
        served = int(dep.sum())
        return {
            "avg_wait": avg_wait,
            "avg_queue": float(q.mean()),
            "max_queue": float(q.max()),
            "avg_travel": self.cfg.free_flow_time + avg_wait,
            "served": float(served),
            "arrived": float(n_arr),
            "throughput": served * 3600.0 / T,
        }

    # ------------------------------------------------------------------
    def simulate(self, controller: TrafficController, arrivals: np.ndarray,
                 engine: str = "fast", with_vehicles: bool = False) -> SimResult:
        """Run one simulation.  engine = 'fast' (vectorised) or 'object' (reference)."""
        T = arrivals.shape[0]
        cap, state = controller.capacity(T)
        if engine == "fast":
            q, dep = self._queues_fast(arrivals, cap)
            metrics = self._metrics(q, dep, arrivals)
            vehicles = None
            if with_vehicles:
                vehicles = []
                for j, a in enumerate(APPROACHES):
                    vehicles += vehicles_from_counts(arrivals[:, j], dep[:, j], a, T)
        elif engine == "object":
            q, vehicles = run_object_engine(arrivals, cap)
            dep = np.zeros_like(q)
            for v in vehicles:
                if v.served:
                    dep[v.departure_time, APPROACHES.index(v.approach)] += 1
            # metrics computed independently from the individual vehicles
            n = len(vehicles)
            total_wait = sum(v.waiting_time(T) for v in vehicles)
            avg_wait = total_wait / n if n else 0.0
            served = sum(v.served for v in vehicles)
            metrics = {"avg_wait": avg_wait, "avg_queue": float(q.mean()),
                       "max_queue": float(q.max()),
                       "avg_travel": self.cfg.free_flow_time + avg_wait,
                       "served": float(served), "arrived": float(n),
                       "throughput": served * 3600.0 / T}
        else:
            raise ValueError("engine must be 'fast' or 'object'")
        return SimResult(metrics, q, dep, arrivals, state, vehicles)


def average_metrics(list_of_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric over several simulation runs (different seeds)."""
    keys = list_of_metrics[0].keys()
    return {k: float(np.mean([m[k] for m in list_of_metrics])) for k in keys}
