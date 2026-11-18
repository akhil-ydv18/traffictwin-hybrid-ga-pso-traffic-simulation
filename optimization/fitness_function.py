"""
fitness_function.py
===================
The objective function J(x) that all optimizers MINIMISE.

    J(x) = w1*W/W_ref + w2*Q/Q_ref + w3*T/T_ref + w4*Qmax/Qmax_ref + P(x)

* W, Q, T, Qmax = average waiting time, average queue, average travel time and
  maximum queue obtained by simulating the signal plan x.
* *_ref = the same metric for the fixed-time baseline on the same traffic.
  Dividing by it makes all terms unit-less, so J = 1.0 means "as good as the
  baseline" and J < 1 means better.
* w1..w4 are normalised to sum to 1.
* P(x) = penalty_coeff * (total violation of bounds + cycle-length limit).

Direction: the algorithms MINIMISE J directly (lower is better).  For plots
where "higher is better" is more natural use fitness_value(J) = 1 / (1 + J).

Randomness: the simulator is stochastic.  The objective is evaluated on a FIXED
list of seeds (common random numbers) so two candidate plans are always
compared on the identical traffic.  Optimisation uses TRAIN seeds; the final
report uses different TEST seeds.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from simulation.traffic_controller import FixedTimeController, TrafficController
from simulation.traffic_simulator import (SimConfig, SimResult, TrafficSimulator,
                                          average_metrics)

METRIC_KEYS = ("avg_wait", "avg_queue", "max_queue", "avg_travel")


@dataclass
class Bounds:
    g_min: float = 10.0    # minimum green (s)
    g_max: float = 60.0    # maximum green (s)
    c_max: float = 120.0   # maximum cycle length (s)

    def __post_init__(self):
        if not 1 <= self.g_min < self.g_max:
            raise ValueError("need 1 <= g_min < g_max")
        if self.c_max <= 0:
            raise ValueError("c_max must be positive")


def normalise_weights(weights: Sequence[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.shape != (4,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be 4 non-negative numbers with a positive sum")
    return w / w.sum()


def fitness_value(J: float) -> float:
    """Transformed 'higher is better' fitness in (0, 1]."""
    return 1.0 / (1.0 + J)


class TrafficObjective:
    """Callable objective: objective(x) -> J.  x = [gNS_1, gEW_1, ..., gNS_K, gEW_K]."""

    def __init__(self, demand: np.ndarray, config: SimConfig, seeds: Sequence[int],
                 n_blocks: int = 3, weights: Sequence[float] = (0.35, 0.25, 0.15, 0.25),
                 bounds: Bounds = Bounds(), penalty_coeff: float = 0.05,
                 baseline_greens=(30, 30)):
        if n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")
        if len(seeds) < 1:
            raise ValueError("need at least one seed")
        self.cfg, self.bounds, self.K = config, bounds, int(n_blocks)
        self.timing = config.timing
        self.weights = normalise_weights(weights)
        self.penalty_coeff = float(penalty_coeff)
        self.seeds = list(seeds)
        self.sim = TrafficSimulator(config)
        self.demand = demand
        # traffic is generated ONCE per seed and reused (common random numbers)
        self.arrivals = [self.sim.generate_arrivals(demand, s) for s in self.seeds]
        self.baseline_controller = FixedTimeController(*baseline_greens, timing=self.timing)
        self.baseline_metrics = self._avg_metrics(self.baseline_controller)
        self.ref = {k: max(self.baseline_metrics[k], 1e-3) for k in METRIC_KEYS}
        self.n_evals = 0

    # ---- search-space description ---------------------------------------
    @property
    def dim(self) -> int:
        return 2 * self.K

    @property
    def lower(self) -> np.ndarray:
        return np.full(self.dim, self.bounds.g_min)

    @property
    def upper(self) -> np.ndarray:
        return np.full(self.dim, self.bounds.g_max)

    # ---- simulation helpers ---------------------------------------------
    def _avg_metrics(self, controller: TrafficController) -> Dict[str, float]:
        cap_state = controller.capacity(self.cfg.duration)[0]    # signal timeline built once
        ms = []
        for arr in self.arrivals:
            q, dep = self.sim._queues_fast(arr, cap_state)
            ms.append(self.sim._metrics(q, dep, arr))
        return average_metrics(ms)

    def controller_for(self, x) -> TrafficController:
        x = np.clip(np.asarray(x, dtype=float), self.lower, self.upper)
        return TrafficController.from_vector(x, self.timing)

    def metrics(self, x) -> Dict[str, float]:
        """Average metrics of plan x over this objective's seeds."""
        return self._avg_metrics(self.controller_for(x))

    def simulate_detailed(self, x, seed_index: int = 0, with_vehicles=False) -> SimResult:
        return self.sim.simulate(self.controller_for(x), self.arrivals[seed_index],
                                 with_vehicles=with_vehicles)

    # ---- the objective ----------------------------------------------------
    def penalty(self, x) -> float:
        x = np.asarray(x, dtype=float)
        b = self.bounds
        v = np.maximum(0, b.g_min - x).sum() + np.maximum(0, x - b.g_max).sum()
        g = x.reshape(-1, 2)
        cycles = g.sum(axis=1) + self.timing.lost_per_cycle()
        v += np.maximum(0, cycles - b.c_max).sum()          # excessive cycle duration
        return self.penalty_coeff * float(v)

    def cost_from_metrics(self, m: Dict[str, float]) -> float:
        terms = np.array([m[k] / self.ref[k] for k in METRIC_KEYS])
        return float(np.dot(self.weights, terms))

    def __call__(self, x) -> float:
        self.n_evals += 1
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,) or not np.all(np.isfinite(x)):
            raise ValueError(f"x must be a finite vector of length {self.dim}")
        # penalty uses the RAW x; the simulation uses x clipped into the box
        return self.cost_from_metrics(self.metrics(x)) + self.penalty(x)

    def cost_of_plan(self, x) -> float:
        """Objective value without incrementing the evaluation counter."""
        n = self.n_evals
        j = self(x)
        self.n_evals = n
        return j
