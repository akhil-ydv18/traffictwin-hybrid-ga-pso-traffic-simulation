"""
evaluation.py - optimizer factory, fair evaluation on unseen test traffic,
percentage improvement, mean +- std summaries, statistical tests.
"""
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from optimization import GeneticAlgorithm, HybridGAPSO, ParticleSwarm

ALGORITHMS = ["GA", "PSO", "Hybrid GA-PSO"]
BASELINE = "Fixed-Time"

# metric -> True if a LARGER value is better
METRICS = {"avg_wait": False, "avg_queue": False, "max_queue": False,
           "avg_travel": False, "served": True, "throughput": True}

DEFAULT_PARAMS = {
    "ga": dict(pop_size=20, crossover_rate=0.9, mutation_rate=0.25, mutation_sigma=0.1,
               tournament_size=3, elite_size=2),
    "pso": dict(swarm_size=20, w_start=0.9, w_end=0.4, c1=1.8, c2=1.8),
    "hybrid": dict(pop_size=20, warmup_fraction=0.3, transfer_fraction=0.5,
                   inject_period=5, inject_fraction=0.25, w_start=0.8, w_end=0.4,
                   c1=1.8, c2=1.8, crossover_rate=0.9, mutation_rate=0.25,
                   mutation_sigma=0.1, tournament_size=3, elite_size=2),
}


def build_optimizer(name: str, objective, budget: int, seed: int, params: Dict = None,
                    patience=None):
    """Create an optimizer.  ALL algorithms get the same evaluation budget
    (`budget` objective calls) so the comparison is fair."""
    p = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
    for k, v in (params or {}).items():
        p[k].update(v)
    lo, hi = objective.lower, objective.upper
    if name == "GA":
        return GeneticAlgorithm(objective, lo, hi, generations=10 ** 6, max_evals=budget,
                                patience=patience, seed=seed, **p["ga"])
    if name == "PSO":
        return ParticleSwarm(objective, lo, hi, iterations=10 ** 6, max_evals=budget,
                             patience=patience, seed=seed, **p["pso"])
    if name == "Hybrid GA-PSO":
        return HybridGAPSO(objective, lo, hi, max_evals=budget, patience=patience, seed=seed, **p["hybrid"])
    if name == "Hybrid (no injection)":
        q = dict(p["hybrid"]); q["inject_period"] = 0
        return HybridGAPSO(objective, lo, hi, max_evals=budget, patience=patience, seed=seed,
                           label=name, **q)
    if name == "Hybrid (no warm-up)":
        q = dict(p["hybrid"]); q["warmup_fraction"] = 0.0
        return HybridGAPSO(objective, lo, hi, max_evals=budget, patience=patience, seed=seed,
                           label=name, **q)
    raise ValueError(f"unknown algorithm '{name}'")


def evaluate_plan(test_objective, x) -> Dict[str, float]:
    """Metrics + objective of plan x on the TEST seeds (never seen by the optimizer)."""
    m = test_objective.metrics(x)
    m["objective"] = test_objective.cost_of_plan(x)
    return m


def baseline_row(test_objective) -> Dict[str, float]:
    m = dict(test_objective.baseline_metrics)
    m["objective"] = test_objective.cost_from_metrics(m)
    return m


def percent_improvement(baseline: float, value: float, higher_is_better: bool) -> float:
    """Positive = better than the baseline."""
    if abs(baseline) < 1e-12:
        return 0.0
    d = (value - baseline) if higher_is_better else (baseline - value)
    return 100.0 * d / abs(baseline)


def summarize(df: pd.DataFrame, group=("scenario", "algorithm")) -> pd.DataFrame:
    """mean and std (over independent runs) of every metric."""
    cols = list(METRICS) + ["objective", "runtime_s", "evals"]
    g = df.groupby(list(group), sort=False)[cols]
    mean, std = g.mean().add_suffix("_mean"), g.std(ddof=1).fillna(0.0).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def improvement_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scn, sub in summary.groupby("scenario", sort=False):
        base = sub[sub.algorithm == BASELINE].iloc[0]
        for _, r in sub.iterrows():
            row = {"scenario": scn, "algorithm": r.algorithm}
            for m, hb in METRICS.items():
                row[f"{m}_impr_%"] = percent_improvement(base[f"{m}_mean"], r[f"{m}_mean"], hb)
            row["objective_impr_%"] = percent_improvement(base["objective_mean"], r["objective_mean"], False)
            rows.append(row)
    return pd.DataFrame(rows)


def resample_history(eval_hist, grid: Sequence[int]) -> List[float]:
    """Best-so-far value at each evaluation count in `grid` (step function)."""
    evals = np.array([e for e, _ in eval_hist])
    best = np.array([b for _, b in eval_hist])
    out = []
    for g in grid:
        idx = np.searchsorted(evals, g, side="right") - 1
        out.append(float(best[max(idx, 0)]))
    return out


def significance(df: pd.DataFrame, reference="Hybrid GA-PSO", column="objective") -> pd.DataFrame:
    """Mann-Whitney U test of `reference` against every other optimizer."""
    from scipy.stats import mannwhitneyu
    rows = []
    for scn, sub in df.groupby("scenario", sort=False):
        ref = sub[sub.algorithm == reference][column].values
        for alg, s2 in sub.groupby("algorithm", sort=False):
            if alg in (reference, BASELINE) or len(s2) < 3 or len(ref) < 3:
                continue
            other = s2[column].values
            try:
                p = float(mannwhitneyu(ref, other, alternative="two-sided").pvalue)
            except ValueError:                       # all values identical
                p = 1.0
            rows.append({"scenario": scn, "reference": reference, "versus": alg,
                         "ref_mean": ref.mean(), "other_mean": other.mean(),
                         "p_value": p, "significant_at_0.05": p < 0.05})
    return pd.DataFrame(rows)
