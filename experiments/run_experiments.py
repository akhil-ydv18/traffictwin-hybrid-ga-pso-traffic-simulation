"""
run_experiments.py - complete experimental study.

    python -m experiments.run_experiments              (full study, a few minutes)
    python -m experiments.run_experiments --quick      (small smoke test, seconds)

Everything is seeded, so re-running gives identical numbers (except run times).
Outputs (all produced by actually running the optimisers):
    data/experiment_results.csv   one row per (scenario, algorithm, run)
    data/summary_mean_std.csv     mean and std over runs
    data/improvement_vs_baseline.csv
    data/significance.csv         Mann-Whitney tests: Hybrid vs GA / PSO
    data/convergence.csv          best-so-far J versus evaluations
    data/popsize_study.csv, data/ablation.csv, data/dimension_study.csv
    data/sample_results.csv       compact summary table (same as the main comparison)
    docs/figures/*.png            charts for the report
    docs/results_summary.md       auto-generated result tables
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from experiments.evaluation import (ALGORITHMS, BASELINE, METRICS, baseline_row, build_optimizer,
                                    evaluate_plan, improvement_table, resample_history,
                                    significance, summarize)
from optimization import TrafficObjective
from simulation import SCENARIOS, SimConfig, scenario_demand
from visualization import performance_charts as pc

TRAIN_SEEDS = (11, 12, 13)                 # traffic seen by the optimiser
TEST_SEEDS = (101, 102, 103, 104, 105)     # unseen traffic used for ALL reported numbers
DATA, FIG = ROOT / "data", ROOT / "docs" / "figures"


def make_objective(scenario, seeds, K=3, duration=3600):
    cfg = SimConfig(duration=duration)
    return TrafficObjective(scenario_demand(scenario, duration), cfg, seeds, n_blocks=K)


def run_one(alg, train_obj, test_obj, budget, run, params=None):
    opt = build_optimizer(alg, train_obj, budget, seed=7919 * run + 17, params=params)
    res = opt.run()
    row = evaluate_plan(test_obj, res.best_x)
    row.update(algorithm=alg, run=run, train_objective=res.best_f, runtime_s=res.runtime,
               evals=res.n_evals, plan="/".join(str(int(round(v))) for v in res.best_x))
    return row, res


def main_comparison(runs, budget, K=3, duration=3600):
    rows, conv, plans = [], [], {}
    grid = np.linspace(0, budget, 41).astype(int)
    for scn in SCENARIOS:
        train, test = make_objective(scn, TRAIN_SEEDS, K, duration), make_objective(scn, TEST_SEEDS, K, duration)
        b = baseline_row(test)
        b.update(scenario=scn, algorithm=BASELINE, run=0, train_objective=1.0, runtime_s=0.0, evals=0, plan="30/30")
        rows.append(b)
        for alg in ALGORITHMS:
            for r in range(runs):
                row, res = run_one(alg, train, test, budget, r)
                row["scenario"] = scn
                rows.append(row)
                for g, v in zip(grid, resample_history(res.eval_history, grid)):
                    conv.append(dict(scenario=scn, algorithm=alg, run=r, evals=int(g), best_f=v))
                plans[(scn, alg, r)] = res.best_x
            print(f"  [{scn:6s}] {alg:14s} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(conv), plans


def study(scenarios, algs, runs, budget, K=3, params_list=None, tag="", duration=3600):
    rows = []
    for scn in scenarios:
        train, test = make_objective(scn, TRAIN_SEEDS, K, duration), make_objective(scn, TEST_SEEDS, K, duration)
        for alg in algs:
            for pl_tag, params in (params_list or [("", None)]):
                for r in range(runs):
                    row, _ = run_one(alg, train, test, budget, r, params)
                    row.update(scenario=scn, variant=pl_tag, K=K)
                    rows.append(row)
    return pd.DataFrame(rows)


def md_table(df, cols, fmt="{:.2f}"):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c])
                                    for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--budget", type=int, default=1000, help="objective evaluations per optimiser run")
    ap.add_argument("--quick", action="store_true", help="tiny run for a smoke test")
    args = ap.parse_args()
    runs, budget = (2, 200) if args.quick else (args.runs, args.budget)
    sub_runs = 2 if args.quick else 8
    DATA.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"== Main comparison ({runs} runs, budget {budget} evals) ==", flush=True)
    df, conv, plans = main_comparison(runs, budget)
    summary = summarize(df)
    impr = improvement_table(summary)
    sig = significance(df)
    df.to_csv(DATA / "experiment_results.csv", index=False)
    summary.to_csv(DATA / "summary_mean_std.csv", index=False)
    impr.to_csv(DATA / "improvement_vs_baseline.csv", index=False)
    sig.to_csv(DATA / "significance.csv", index=False)
    conv.to_csv(DATA / "convergence.csv", index=False)
    summary.merge(impr, on=["scenario", "algorithm"]).to_csv(DATA / "sample_results.csv", index=False)

    print("== Population / swarm size study (Peak) ==", flush=True)
    sizes = [("size=10", dict(ga=dict(pop_size=10), pso=dict(swarm_size=10), hybrid=dict(pop_size=10))),
             ("size=20", dict(ga=dict(pop_size=20), pso=dict(swarm_size=20), hybrid=dict(pop_size=20))),
             ("size=40", dict(ga=dict(pop_size=40), pso=dict(swarm_size=40), hybrid=dict(pop_size=40)))]
    pop = study(["Peak"], ALGORITHMS, sub_runs, budget, params_list=sizes)
    pop.to_csv(DATA / "popsize_study.csv", index=False)

    print("== Ablation (High, Peak) ==", flush=True)
    abl = study(["High", "Peak"], ALGORITHMS + ["Hybrid (no injection)", "Hybrid (no warm-up)"], runs, budget)
    abl.to_csv(DATA / "ablation.csv", index=False)

    print("== Dimension study (Peak, K = 1, 3, 6) ==", flush=True)
    dim = pd.concat([study(["Peak"], ALGORITHMS, sub_runs, budget, K=K) for K in (1, 3, 6)])
    dim.to_csv(DATA / "dimension_study.csv", index=False)

    # ----- figures ---------------------------------------------------------
    for scn in SCENARIOS:
        pc.save_convergence(conv, scn, FIG / f"convergence_{scn.lower()}.png")
    for m in ("avg_wait", "avg_queue", "max_queue", "throughput", "avg_travel"):
        pc.save_metric_bars(summary, m, FIG / f"bars_{m}.png")
    for scn in ("Medium", "Peak"):
        pc.save_before_after(summary, scn, "Hybrid GA-PSO", FIG / f"before_after_{scn.lower()}.png")
    abl_sum = summarize(abl)
    for scn in ("High", "Peak"):
        pc.save_ablation(abl_sum, scn, FIG / f"ablation_{scn.lower()}.png")
    test = make_objective("Peak", TEST_SEEDS)
    best_h = plans[("Peak", "Hybrid GA-PSO", 0)]
    series = {BASELINE: test.sim.simulate(test.baseline_controller, test.arrivals[0]).queue_ts,
              "Hybrid GA-PSO": test.simulate_detailed(best_h, 0).queue_ts}
    pc.save_density(series, FIG / "density_peak.png", "Traffic density over time - Peak traffic (test seed 101)")

    # ----- markdown tables -------------------------------------------------
    s = summary.copy()
    for m in ("avg_wait", "avg_queue", "max_queue", "avg_travel", "throughput", "objective"):
        s[m] = s[f"{m}_mean"].map("{:.2f}".format) + " +- " + s[f"{m}_std"].map("{:.2f}".format)
    s["served"] = s["served_mean"].map("{:.0f}".format)
    s["opt_time_s"] = s["runtime_s_mean"].map("{:.2f}".format)
    ab = abl_sum.copy(); ab["J (mean +- std)"] = ab["objective_mean"].map("{:.4f}".format) + " +- " + ab["objective_std"].map("{:.4f}".format)
    ps = summarize(pop, ("variant", "algorithm")); ps["J (mean +- std)"] = ps["objective_mean"].map("{:.4f}".format) + " +- " + ps["objective_std"].map("{:.4f}".format)
    ds = summarize(dim, ("K", "algorithm")); ds["J (mean +- std)"] = ds["objective_mean"].map("{:.4f}".format) + " +- " + ds["objective_std"].map("{:.4f}".format)
    imp = impr.copy()
    md = [f"# Auto-generated results\n\nGenerated by `experiments/run_experiments.py`: {runs} independent runs per "
          f"cell, budget {budget} evaluations per run, train seeds {TRAIN_SEEDS}, test seeds {TEST_SEEDS}. "
          "All values are measured on the unseen TEST seeds. Mean +- std over runs.\n",
          "## Main comparison\n", md_table(s, ["scenario", "algorithm", "avg_wait", "avg_queue", "max_queue",
                                              "avg_travel", "served", "throughput", "objective", "opt_time_s"]),
          "\n## Percentage improvement over Fixed-Time (positive = better)\n",
          md_table(imp, ["scenario", "algorithm", "avg_wait_impr_%", "avg_queue_impr_%", "max_queue_impr_%",
                         "avg_travel_impr_%", "throughput_impr_%", "objective_impr_%"], "{:.1f}"),
          "\n## Significance (Mann-Whitney U, Hybrid vs others, objective J)\n",
          md_table(sig, ["scenario", "versus", "ref_mean", "other_mean", "p_value", "significant_at_0.05"], "{:.4f}") if len(sig) else "n/a",
          "\n## Ablation\n", md_table(ab, ["scenario", "algorithm", "J (mean +- std)"]),
          "\n## Population / swarm size (Peak)\n", md_table(ps, ["variant", "algorithm", "J (mean +- std)"]),
          "\n## Dimension study (Peak): K blocks -> 2K variables\n", md_table(ds, ["K", "algorithm", "J (mean +- std)"])]
    (ROOT / "docs" / "results_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"All done in {time.time() - t0:.0f} s.  Results in data/ and docs/figures/")


if __name__ == "__main__":
    main()
