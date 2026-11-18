"""
hybrid_ga_pso.py - the Hybrid GA-PSO optimizer (the core of the project).

Why hybridise?
  * GA is a good EXPLORER: crossover/mutation sample widely, so it rarely misses
    the promising region, but it fine-tunes slowly.
  * PSO is a good EXPLOITER: particles are pulled towards the best positions and
    converge fast, but the swarm can collapse early onto a local optimum.

Pipeline (information flows in BOTH directions between the two methods)
  Phase 1  GA warm-up        : evolve a population for ~warmup_fraction of the budget.
  Phase 2  GA -> PSO transfer : the best `transfer_fraction` of the GA population
                                become particles (their fitness is already known, so
                                the transfer costs no extra evaluations); the other
                                particles are fresh random ones for diversity.
  Phase 3  PSO refinement     : normal PSO iterations (pbest / gbest / inertia).
  Phase 4  PSO -> GA injection: every `inject_period` iterations the personal bests are
                                used as GA parents (tournament selection, crossover,
                                mutation) and the offspring REPLACE the worst particles.
                                This restores diversity and lets PSO escape stagnation.
  Stop     when the evaluation budget is used up or gbest has not improved for
           `patience` iterations.
Setting warmup_fraction=0 or inject_period=0 turns off a component (ablation study).
"""
import time
from typing import Optional

import numpy as np

from .common import OptResult, check_bounds, diversity
from .genetic_algorithm import GeneticAlgorithm
from .particle_swarm import ParticleSwarm


class HybridGAPSO:
    name = "Hybrid GA-PSO"

    def __init__(self, objective, lower, upper, pop_size=20, max_evals=800,
                 warmup_fraction=0.3, transfer_fraction=0.5,
                 inject_period=5, inject_fraction=0.25,
                 w_start=0.8, w_end=0.4, c1=1.8, c2=1.8, vmax_frac=0.25,
                 crossover_rate=0.9, mutation_rate=0.25, mutation_sigma=0.1,
                 tournament_size=3, elite_size=2, blx_alpha=0.3,
                 patience: Optional[int] = None, seed=0, label=None):
        self.lower, self.upper = check_bounds(lower, upper)
        if pop_size < 4:
            raise ValueError("pop_size must be >= 4")
        if max_evals < 2 * pop_size:
            raise ValueError("max_evals must be at least 2 * pop_size")
        if not 0 <= warmup_fraction < 1 or not 0 < transfer_fraction <= 1:
            raise ValueError("warmup_fraction in [0,1), transfer_fraction in (0,1]")
        if inject_period < 0 or not 0 <= inject_fraction <= 1:
            raise ValueError("inject_period >= 0 and inject_fraction in [0,1]")
        if label:
            self.name = label
        self.f = objective
        self.n, self.max_evals = int(pop_size), int(max_evals)
        self.warm, self.transfer = warmup_fraction, transfer_fraction
        self.period, self.inj_frac = int(inject_period), inject_fraction
        self.patience, self.seed = patience, seed
        self.pso_kw = dict(w_start=w_start, w_end=w_end, c1=c1, c2=c2, vmax_frac=vmax_frac)
        self.ga_kw = dict(crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                          mutation_sigma=mutation_sigma, tournament_size=tournament_size,
                          elite_size=elite_size, blx_alpha=blx_alpha)

    def run(self) -> OptResult:
        t0 = time.perf_counter()
        lo, hi = self.lower, self.upper
        rng = np.random.default_rng(self.seed + 12345)
        n_used = 0
        hist, ehist, div = [], [], []

        # ---- Phase 1: GA warm-up (global exploration) -----------------------
        ga_budget = max(self.n, int(self.warm * self.max_evals)) if self.warm > 0 else 0
        ga = GeneticAlgorithm(self.f, lo, hi, pop_size=self.n, generations=10 ** 6,
                              max_evals=max(ga_budget, self.n), seed=self.seed, **self.ga_kw)
        if ga_budget > 0:
            g = ga.run()
            n_used += g.n_evals
            hist += g.history
            ehist += g.eval_history
            div += g.extra["diversity"]
            pop, fit = g.extra["population"], g.extra["fitness"]
            k = max(1, int(np.ceil(self.transfer * self.n)))
            top = np.argsort(fit)[:k]
            seeds_pos, seeds_fit = pop[top], fit[top]
        else:
            seeds_pos, seeds_fit = np.empty((0, len(lo))), np.empty(0)
        warm_len = len(hist)

        # ---- Phase 2: build the swarm (GA elites + random newcomers) --------
        n_rand = self.n - len(seeds_pos)
        rand_pos = rng.uniform(lo, hi, (n_rand, len(lo)))
        rand_fit = np.array([float(self.f(p)) for p in rand_pos])
        n_used += n_rand
        pso = ParticleSwarm(self.f, lo, hi, swarm_size=self.n, **self.pso_kw, seed=self.seed + 7)
        pso.initialize(np.vstack([seeds_pos, rand_pos]), np.concatenate([seeds_fit, rand_fit]))
        hist.append(pso.gbest_fit)
        ehist.append((n_used, pso.gbest_fit))
        div.append(diversity(pso.pos, lo, hi))

        # ---- Phase 3 + 4: PSO refinement with periodic GA injection ---------
        n_rep = max(1, int(round(self.inj_frac * self.n))) if (self.period and self.inj_frac > 0) else 0
        start_used, injections, it = n_used, 0, 0
        stall, prev = 0, pso.gbest_fit
        while True:
            it += 1
            will_inject = n_rep > 0 and it % self.period == 0
            cost = self.n + (n_rep if will_inject else 0)
            if n_used + cost > self.max_evals:
                break                                              # budget exhausted
            progress = (n_used - start_used) / max(1, self.max_evals - start_used)
            e0 = pso.n_evals
            pso.step(pso.inertia(progress))
            n_used += pso.n_evals - e0
            if will_inject:                                        # GA operators on personal bests
                kids = ga.make_offspring(pso.pbest_pos, pso.pbest_fit, n_rep)
                kid_fit = np.array([float(self.f(c)) for c in kids])
                n_used += n_rep
                worst = np.argsort(pso.pbest_fit)[-n_rep:]         # replace the worst particles
                pso.pos[worst], pso.pbest_pos[worst] = kids, kids
                pso.pbest_fit[worst] = kid_fit
                pso.vel[worst] = rng.uniform(-0.1, 0.1, kids.shape) * (hi - lo)
                pso._update_gbest()
                injections += 1
            hist.append(pso.gbest_fit)
            ehist.append((n_used, pso.gbest_fit))
            div.append(diversity(pso.pos, lo, hi))
            stall = 0 if pso.gbest_fit < prev - 1e-9 else stall + 1
            prev = min(prev, pso.gbest_fit)
            if self.patience and stall >= self.patience:           # stagnation stop
                break
        return OptResult(self.name, pso.gbest_pos.copy(), pso.gbest_fit, hist, ehist, n_used,
                         time.perf_counter() - t0,
                         {"diversity": div, "warmup_len": warm_len, "injections": injections,
                          "warmup_evals": ga_budget and ga.n_evals})
