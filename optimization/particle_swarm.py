"""
particle_swarm.py - Particle Swarm Optimization written from scratch.

Each particle i has a position x_i (a signal plan), a velocity v_i, a personal
best pbest_i and the swarm shares a global best gbest.

    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)        r1, r2 ~ U(0,1)
    x <- x + v

w  (inertia)  : memory of the old velocity, decreased linearly w_start -> w_end
                (large w = explore, small w = exploit).
c1 (cognitive): pull towards the particle's own best.
c2 (social)   : pull towards the swarm's best.
Boundaries    : positions are clamped into [lower, upper] and the velocity
                component that hit the wall is set to zero; velocities are
                clamped to +-vmax_frac * range.
"""
import time
from typing import Optional

import numpy as np

from .common import OptResult, check_bounds, diversity


class ParticleSwarm:
    name = "PSO"

    def __init__(self, objective, lower, upper, swarm_size=20, iterations=50,
                 w_start=0.9, w_end=0.4, c1=1.8, c2=1.8, vmax_frac=0.25,
                 max_evals: Optional[int] = None, patience: Optional[int] = None, seed=0):
        self.lower, self.upper = check_bounds(lower, upper)
        if swarm_size < 2:
            raise ValueError("swarm_size must be >= 2")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if min(w_start, w_end, c1, c2) < 0 or vmax_frac <= 0:
            raise ValueError("w, c1, c2 must be >= 0 and vmax_frac > 0")
        if max_evals is not None and max_evals < swarm_size:
            raise ValueError("max_evals must be >= swarm_size")
        self.f = objective
        self.n, self.iterations = int(swarm_size), int(iterations)
        self.w_start, self.w_end, self.c1, self.c2 = w_start, w_end, c1, c2
        self.max_evals, self.patience = max_evals, patience
        self.rng = np.random.default_rng(seed)
        self.dim = self.lower.size
        self.vmax = vmax_frac * (self.upper - self.lower)
        self.n_evals = 0

    def _eval(self, x) -> float:
        self.n_evals += 1
        return float(self.f(x))

    def initialize(self, positions=None, fitness=None):
        """Create the swarm.  `positions`/`fitness` may be supplied (the hybrid
        passes in GA individuals whose fitness is already known)."""
        if positions is None:
            positions = self.rng.uniform(self.lower, self.upper, (self.n, self.dim))
        self.pos = np.array(positions, dtype=float)
        self.n = len(self.pos)
        self.vel = self.rng.uniform(-0.1, 0.1, self.pos.shape) * (self.upper - self.lower)
        if fitness is None:
            fitness = [self._eval(p) for p in self.pos]
        self.pbest_pos = self.pos.copy()
        self.pbest_fit = np.array(fitness, dtype=float)
        self._update_gbest()

    def _update_gbest(self):
        i = int(np.argmin(self.pbest_fit))
        self.gbest_pos, self.gbest_fit = self.pbest_pos[i].copy(), float(self.pbest_fit[i])

    def step(self, w: float):
        """One PSO iteration (costs one evaluation per particle)."""
        r1 = self.rng.random(self.pos.shape)
        r2 = self.rng.random(self.pos.shape)
        self.vel = (w * self.vel
                    + self.c1 * r1 * (self.pbest_pos - self.pos)      # cognitive term
                    + self.c2 * r2 * (self.gbest_pos - self.pos))     # social term
        self.vel = np.clip(self.vel, -self.vmax, self.vmax)
        new = self.pos + self.vel
        hit = (new < self.lower) | (new > self.upper)                 # boundary constraint
        self.pos = np.clip(new, self.lower, self.upper)
        self.vel[hit] = 0.0
        fit = np.array([self._eval(p) for p in self.pos])
        better = fit < self.pbest_fit                                 # update pbest
        self.pbest_pos[better] = self.pos[better]
        self.pbest_fit[better] = fit[better]
        self._update_gbest()                                          # update gbest

    def inertia(self, progress: float) -> float:
        return self.w_start + (self.w_end - self.w_start) * min(1.0, max(0.0, progress))

    def run(self) -> OptResult:
        t0 = time.perf_counter()
        self.initialize()
        iters = self.iterations
        if self.max_evals is not None:
            iters = min(iters, 1 + (self.max_evals - self.n_evals) // self.n)
        history, eval_hist = [self.gbest_fit], [(self.n_evals, self.gbest_fit)]
        div = [diversity(self.pos, self.lower, self.upper)]
        stall, prev = 0, self.gbest_fit
        for it in range(1, iters):
            self.step(self.inertia(it / max(1, iters - 1)))
            history.append(self.gbest_fit)
            eval_hist.append((self.n_evals, self.gbest_fit))
            div.append(diversity(self.pos, self.lower, self.upper))
            stall = 0 if self.gbest_fit < prev - 1e-9 else stall + 1
            prev = min(prev, self.gbest_fit)
            if self.patience and stall >= self.patience:
                break
        return OptResult(self.name, self.gbest_pos.copy(), self.gbest_fit, history, eval_hist,
                         self.n_evals, time.perf_counter() - t0, {"diversity": div})
