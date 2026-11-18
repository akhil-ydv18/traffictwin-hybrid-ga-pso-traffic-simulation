"""
genetic_algorithm.py - Genetic Algorithm written from scratch (real-coded).

Chromosome : real vector [gNS_1, gEW_1, ..., gNS_K, gEW_K] (seconds).
Selection  : tournament (pick k random individuals, the lowest J wins).
Crossover  : BLX-alpha blend crossover - each child gene is drawn uniformly from
             the interval spanned by the parents' genes, widened by alpha.
Mutation   : per-gene Gaussian noise, sigma = fraction of the variable range.
Elitism    : the best `elite_size` individuals are copied unchanged.
Fitness    : J(x) from fitness_function (MINIMISED).
"""
import time
from typing import Optional

import numpy as np

from .common import OptResult, check_bounds, diversity


class GeneticAlgorithm:
    name = "GA"

    def __init__(self, objective, lower, upper, pop_size=20, generations=50,
                 crossover_rate=0.9, mutation_rate=0.25, mutation_sigma=0.1,
                 tournament_size=3, elite_size=2, blx_alpha=0.3,
                 max_evals: Optional[int] = None, patience: Optional[int] = None, seed=0):
        self.lower, self.upper = check_bounds(lower, upper)
        if pop_size < 4:
            raise ValueError("pop_size must be >= 4")
        if generations < 1:
            raise ValueError("generations must be >= 1")
        if not 0 <= crossover_rate <= 1 or not 0 <= mutation_rate <= 1:
            raise ValueError("crossover_rate and mutation_rate must be in [0, 1]")
        if not 1 <= tournament_size <= pop_size:
            raise ValueError("tournament_size must be in [1, pop_size]")
        if not 0 <= elite_size < pop_size:
            raise ValueError("elite_size must be in [0, pop_size)")
        if max_evals is not None and max_evals < pop_size:
            raise ValueError("max_evals must be >= pop_size")
        self.f = objective
        self.pop_size, self.generations = int(pop_size), int(generations)
        self.pc, self.pm, self.sigma = crossover_rate, mutation_rate, mutation_sigma
        self.k, self.elite, self.alpha = int(tournament_size), int(elite_size), blx_alpha
        self.max_evals, self.patience = max_evals, patience
        self.rng = np.random.default_rng(seed)
        self.dim = self.lower.size
        self.n_evals = 0

    # ---- building blocks (also reused by the hybrid) -----------------------
    def _eval(self, x) -> float:
        self.n_evals += 1
        return float(self.f(x))

    def init_population(self) -> np.ndarray:
        """Population initialisation: uniform random inside the bounds."""
        return self.rng.uniform(self.lower, self.upper, (self.pop_size, self.dim))

    def select_parent(self, pop, fit) -> np.ndarray:
        """Tournament selection."""
        idx = self.rng.integers(0, len(pop), self.k)
        return pop[idx[np.argmin(fit[idx])]]

    def crossover(self, a, b) -> np.ndarray:
        """BLX-alpha crossover (applied with probability pc)."""
        if self.rng.random() > self.pc:
            return a.copy()
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        d = hi - lo
        child = self.rng.uniform(lo - self.alpha * d, hi + self.alpha * d)
        return np.clip(child, self.lower, self.upper)

    def mutate(self, x) -> np.ndarray:
        """Gaussian mutation, each gene mutated with probability pm."""
        mask = self.rng.random(self.dim) < self.pm
        noise = self.rng.normal(0.0, self.sigma * (self.upper - self.lower))
        return np.clip(np.where(mask, x + noise, x), self.lower, self.upper)

    def make_offspring(self, pop, fit, n) -> np.ndarray:
        """n children = select two parents -> crossover -> mutation."""
        kids = [self.mutate(self.crossover(self.select_parent(pop, fit),
                                           self.select_parent(pop, fit))) for _ in range(n)]
        return np.array(kids)

    # ---- main loop -----------------------------------------------------------
    def run(self) -> OptResult:
        t0 = time.perf_counter()
        pop = self.init_population()
        fit = np.array([self._eval(x) for x in pop])
        best_i = int(np.argmin(fit))
        best_x, best_f = pop[best_i].copy(), float(fit[best_i])
        history, eval_hist, div = [best_f], [(self.n_evals, best_f)], [diversity(pop, self.lower, self.upper)]
        stall = 0
        n_child = self.pop_size - self.elite
        for _ in range(1, self.generations):
            if self.max_evals is not None and self.n_evals + n_child > self.max_evals:
                break                                           # evaluation budget used up
            order = np.argsort(fit)
            elites, elite_fit = pop[order[:self.elite]], fit[order[:self.elite]]   # elitism
            kids = self.make_offspring(pop, fit, n_child)
            kid_fit = np.array([self._eval(c) for c in kids])
            pop = np.vstack([elites, kids]) if self.elite else kids
            fit = np.concatenate([elite_fit, kid_fit]) if self.elite else kid_fit
            i = int(np.argmin(fit))
            if fit[i] < best_f - 1e-9:
                best_x, best_f, stall = pop[i].copy(), float(fit[i]), 0
            else:
                stall += 1
            history.append(best_f)
            eval_hist.append((self.n_evals, best_f))
            div.append(diversity(pop, self.lower, self.upper))
            if self.patience and stall >= self.patience:        # stagnation stop
                break
        return OptResult(self.name, best_x, best_f, history, eval_hist, self.n_evals,
                         time.perf_counter() - t0,
                         {"population": pop, "fitness": fit, "diversity": div})
