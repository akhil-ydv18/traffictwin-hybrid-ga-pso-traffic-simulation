"""
TrafficTwin - Streamlit dashboard.        Run with:   streamlit run app.py

Workflow: set the traffic and algorithm parameters in the sidebar, press
"Run Simulation" (fixed-time baseline) and/or "Run Optimization" (GA, PSO and
Hybrid GA-PSO are really executed), then explore the tabs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from experiments.evaluation import (ALGORITHMS, BASELINE, METRICS, baseline_row, build_optimizer,
                                    evaluate_plan, percent_improvement)
from optimization import Bounds, TrafficObjective
from simulation import SCENARIOS, SimConfig, build_demand
from simulation.traffic_controller import STATE_NAMES
from visualization import performance_charts as pc
from visualization.traffic_visualization import build_intersection_figure, density_level

TRAIN_SEEDS_ALL = [11, 12, 13, 14, 15]
TEST_SEEDS = [101, 102, 103, 104, 105]
DATA, FIG = ROOT / "data", ROOT / "docs" / "figures"

st.set_page_config(page_title="TrafficTwin", page_icon="🚦", layout="wide")
st.markdown("""
<style>
.hero {background: linear-gradient(90deg,#0f172a,#1e3a8a); padding: 1.1rem 1.5rem; border-radius: 12px; color: white;}
.hero h1 {margin: 0; font-size: 2rem; color: white;} .hero p {margin: .2rem 0 0 0; color: #cbd5e1;}
</style>
<div class="hero"><h1>🚦 TrafficTwin</h1>
<p>Hybrid GA-PSO calibration of traffic-signal timings - Optimization Techniques for Intelligent Systems</p></div>
""", unsafe_allow_html=True)


def read_csv_safe(path):
    """Read a result CSV; return an empty DataFrame if missing, empty or unreadable."""
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def show(fn, *args, **kwargs):
    """Call a Streamlit element so that it stretches to the container width on
    both old and new Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


# =============================== SIDEBAR =====================================
sb = st.sidebar
sb.header("1 · Traffic")
preset = sb.selectbox("Scenario preset", list(SCENARIOS) + ["Custom"], index=2)
base = SCENARIOS.get(preset, SCENARIOS["Medium"])
rates = [sb.slider(f"{n} arrivals (veh/min)", 0.0, 20.0, float(r), 0.5, key=f"rate{n}_{preset}")
         for n, r in zip("NSEW", base["rates"])]
shift = sb.slider("Demand shift between NS and EW", 0.0, 0.6, float(base["shift"]), 0.05, key=f"shift_{preset}")
duration = sb.slider("Simulation duration (s)", 600, 7200, 3600, 300)

sb.header("2 · Signals")
g_ns = sb.slider("Fixed-time baseline: NS green (s)", 5, 90, 30)
g_ew = sb.slider("Fixed-time baseline: EW green (s)", 5, 90, 30)
K = sb.slider("Time blocks K (2K variables)", 1, 6, 3)
with sb.expander("Constraints and objective weights"):
    g_min = st.number_input("Minimum green (s)", 5, 40, 10)
    g_max = st.number_input("Maximum green (s)", 20, 120, 60)
    c_max = st.number_input("Maximum cycle (s)", 40, 240, 120)
    w_raw = [st.slider("w1 waiting time", 0.0, 1.0, 0.35, 0.05),
             st.slider("w2 average queue", 0.0, 1.0, 0.25, 0.05),
             st.slider("w3 travel time", 0.0, 1.0, 0.15, 0.05),
             st.slider("w4 maximum queue", 0.0, 1.0, 0.25, 0.05)]

sb.header("3 · Optimizers")
budget = sb.slider("Evaluation budget per algorithm", 200, 5000, 1000, 100)
n_train = sb.slider("Training traffic seeds", 1, 5, 3)
master_seed = sb.number_input("Random seed", 0, 10_000, 7)
with sb.expander("Genetic Algorithm"):
    ga_pop = st.slider("Population size", 6, 80, 20, key="ga_pop")
    ga_pc = st.slider("Crossover rate", 0.0, 1.0, 0.9, 0.05)
    ga_pm = st.slider("Mutation rate (per gene)", 0.0, 1.0, 0.25, 0.05)
    ga_sig = st.slider("Mutation sigma (fraction of range)", 0.01, 0.5, 0.10, 0.01)
    ga_k = st.slider("Tournament size", 2, 6, 3)
    ga_el = st.slider("Elite individuals", 0, 5, 2)
with sb.expander("Particle Swarm"):
    pso_n = st.slider("Swarm size", 6, 80, 20, key="pso_n")
    pso_w0 = st.slider("Inertia start", 0.3, 1.0, 0.9, 0.05)
    pso_w1 = st.slider("Inertia end", 0.1, 0.9, 0.4, 0.05)
    pso_c1 = st.slider("Cognitive c1", 0.0, 3.0, 1.8, 0.1)
    pso_c2 = st.slider("Social c2", 0.0, 3.0, 1.8, 0.1)
with sb.expander("Hybrid GA-PSO"):
    hy_n = st.slider("Population / swarm size", 6, 80, 20, key="hy_n")
    hy_warm = st.slider("GA warm-up fraction of budget", 0.05, 0.8, 0.3, 0.05)
    hy_tr = st.slider("Fraction of GA population moved into swarm", 0.1, 1.0, 0.5, 0.1)
    hy_per = st.slider("GA injection every N iterations (0 = off)", 0, 20, 5)
    hy_frac = st.slider("Fraction of swarm replaced at injection", 0.05, 0.6, 0.25, 0.05)
run_sim = sb.button("▶ Run Simulation", type="secondary")
run_opt = sb.button("⚡ Run Optimization", type="primary")

PARAMS = {"ga": dict(pop_size=ga_pop, crossover_rate=ga_pc, mutation_rate=ga_pm, mutation_sigma=ga_sig,
                     tournament_size=min(ga_k, ga_pop), elite_size=min(ga_el, ga_pop - 1)),
          "pso": dict(swarm_size=pso_n, w_start=pso_w0, w_end=pso_w1, c1=pso_c1, c2=pso_c2),
          "hybrid": dict(pop_size=hy_n, warmup_fraction=hy_warm, transfer_fraction=hy_tr,
                         inject_period=hy_per, inject_fraction=hy_frac, crossover_rate=ga_pc,
                         mutation_rate=ga_pm, mutation_sigma=ga_sig, tournament_size=min(ga_k, hy_n),
                         elite_size=min(ga_el, hy_n - 1))}


# ============================ RUN LOGIC =======================================
def build_context():
    """Validate the inputs and build the training and test objectives."""
    if g_min >= g_max:
        raise ValueError("Minimum green must be smaller than maximum green.")
    if sum(w_raw) <= 0:
        raise ValueError("At least one objective weight must be positive.")
    if sum(rates) <= 0:
        raise ValueError("Arrival rates cannot all be zero.")
    cfg = SimConfig(duration=int(duration))
    demand = build_demand(rates, shift, int(duration))
    bounds = Bounds(g_min, g_max, c_max)
    kw = dict(n_blocks=K, weights=w_raw, bounds=bounds, baseline_greens=(g_ns, g_ew))
    return dict(train=TrafficObjective(demand, cfg, TRAIN_SEEDS_ALL[:n_train], **kw),
                test=TrafficObjective(demand, cfg, TEST_SEEDS, **kw), K=K, label=preset)


def fmt_plan(x):
    return " | ".join(f"B{i + 1}: {int(round(a))}/{int(round(b))}" for i, (a, b) in enumerate(np.asarray(x).reshape(-1, 2)))


if run_sim or run_opt:
    try:
        ctx = build_context()
        st.session_state["ctx"] = ctx
    except ValueError as e:
        st.error(f"Invalid input: {e}")
        st.stop()

if run_opt:
    ctx = st.session_state["ctx"]
    min_budget = 2 * max(ga_pop, pso_n, hy_n)
    if budget < min_budget:
        st.error(f"Budget must be at least {min_budget} evaluations for these population sizes.")
        st.stop()
    rows = {BASELINE: dict(baseline_row(ctx["test"]), runtime_s=0.0, evals=0, plan=f"{g_ns}/{g_ew} (fixed)")}
    results = {}
    bar = st.progress(0.0, text="Starting optimisation...")
    for i, alg in enumerate(ALGORITHMS):
        bar.progress(i / len(ALGORITHMS), text=f"Running {alg} ...")
        res = build_optimizer(alg, ctx["train"], budget, int(master_seed), PARAMS).run()
        results[alg] = res
        rows[alg] = dict(evaluate_plan(ctx["test"], res.best_x), runtime_s=res.runtime,
                         evals=res.n_evals, plan=fmt_plan(res.best_x))
    bar.progress(1.0, text="Optimisation finished")
    st.session_state["opt"] = dict(results=results, rows=rows)

ctx, opt = st.session_state.get("ctx"), st.session_state.get("opt")


def simulate_controller(name, seed_idx):
    test = ctx["test"]
    if name == BASELINE:
        return test.sim.simulate(test.baseline_controller, test.arrivals[seed_idx])
    return test.simulate_detailed(opt["results"][name].best_x, seed_idx)


# ================================ TABS =======================================
t_over, t_live, t_opt, t_cmp, t_exp, t_theory = st.tabs(
    ["Overview", "Live intersection", "Optimization", "Comparison", "Saved experiments", "Theory"])

with t_over:
    st.subheader("What this project does")
    st.write("A four-way intersection is simulated second by second. A signal plan assigns green times "
             "to the North-South and East-West streams in each of K time blocks. **GA**, **PSO** and a "
             "**hybrid GA-PSO** search for the plan that minimises a weighted congestion cost, and are "
             "compared with a conventional fixed-time controller on traffic they have never seen.")
    if opt is None:
        st.info("Use the sidebar: **Run Simulation** shows the fixed-time baseline, **Run Optimization** "
                "runs the three optimisers (takes a few seconds).")
    else:
        rows = opt["rows"]
        best = min(ALGORITHMS, key=lambda a: rows[a]["objective"])
        cols = st.columns(4)
        for c, (lab, key, hb) in zip(cols, [("Avg waiting time", "avg_wait", False), ("Avg queue", "avg_queue", False),
                                            ("Max queue", "max_queue", False), ("Throughput (veh/h)", "throughput", True)]):
            imp = percent_improvement(rows[BASELINE][key], rows[best][key], hb)
            c.metric(f"{lab} - {best}", f"{rows[best][key]:.1f}", f"{imp:+.1f}% vs fixed-time")
        st.caption("Values are measured on unseen test traffic (seeds 101-105).")

with t_live:
    if ctx is None:
        st.info("Press **Run Simulation** in the sidebar to start.")
    else:
        names = [BASELINE] + (list(opt["results"]) if opt else [])
        c1, c2, c3 = st.columns(3)
        who = c1.selectbox("Controller", names)
        seed_idx = c2.selectbox("Traffic sample", list(range(len(TEST_SEEDS))), format_func=lambda i: f"seed {TEST_SEEDS[i]}")
        T = ctx["test"].cfg.duration
        t0 = c3.slider("Animation window starts at (s)", 0, max(0, T - 120), 0, 10)
        sim = simulate_controller(who, seed_idx)
        left, right = st.columns([3, 2])
        with left:
            show(st.plotly_chart, build_intersection_figure(sim.queue_ts, sim.dep_ts, sim.state_ts, t0, 120, 2))
            st.caption("Squares = vehicles (blue N, teal S, orange E, purple W). Circles = signals. Press Play.")
        with right:
            st.markdown(f"**Signal at t = {t0} s:** {STATE_NAMES[int(sim.state_ts[t0])]}")
            st.markdown("**Traffic density indicators** (average queue over the whole run)")
            for j, n in enumerate("NSEW"):
                aq = float(sim.queue_ts[:, j].mean())
                st.progress(min(1.0, aq / 15.0), text=f"{n}: {aq:.1f} veh - {density_level(aq)}")
            m = sim.metrics
            a, b = st.columns(2)
            a.metric("Avg wait (s)", f"{m['avg_wait']:.1f}"); b.metric("Avg queue", f"{m['avg_queue']:.2f}")
            a.metric("Max queue", f"{m['max_queue']:.0f}"); b.metric("Throughput (veh/h)", f"{m['throughput']:.0f}")
        show(st.plotly_chart, pc.approach_queue_fig(sim.queue_ts))
        if opt is not None:
            series = {n: simulate_controller(n, seed_idx).queue_ts for n in names}
            show(st.plotly_chart, pc.density_fig(series))

with t_opt:
    if opt is None:
        st.info("Press **Run Optimization** in the sidebar.")
    else:
        res = opt["results"]
        show(st.plotly_chart, pc.convergence_fig(res))
        cols = st.columns(3)
        for c, n in zip(cols, res):
            with c:
                show(st.plotly_chart, pc.convergence_fig(res, show_only=[n]))
        show(st.plotly_chart, pc.diversity_fig(res))
        h = res["Hybrid GA-PSO"].extra
        st.caption(f"Hybrid: GA warm-up lasted {h['warmup_len']} generations; {h['injections']} GA-injection steps "
                   "replaced the worst particles. J = 1 is the fixed-time baseline on the training traffic; lower is better.")

with t_cmp:
    if opt is None:
        st.info("Press **Run Optimization** in the sidebar.")
    else:
        rows = opt["rows"]
        df = pd.DataFrame([{"Controller": n, "Avg wait (s)": r["avg_wait"], "Avg queue": r["avg_queue"],
                            "Max queue": r["max_queue"], "Avg travel (s)": r["avg_travel"],
                            "Vehicles served": r["served"], "Throughput (veh/h)": r["throughput"],
                            "Objective J": r["objective"], "Optimisation time (s)": r["runtime_s"],
                            "Evaluations": r["evals"], "Signal plan (NS/EW green per block)": r["plan"]}
                           for n, r in rows.items()])
        st.subheader("Results on unseen test traffic")
        show(st.dataframe, df.round(3), hide_index=True)
        imp = pd.DataFrame([{"Controller": n, **{f"{k} (%)": percent_improvement(rows[BASELINE][k], r[k], hb)
                                                 for k, hb in METRICS.items()}} for n, r in rows.items() if n != BASELINE])
        st.subheader("Improvement over fixed-time (positive = better)")
        show(st.dataframe, imp.round(1), hide_index=True)
        names = list(rows)
        c1, c2, c3 = st.columns(3)
        with c1:
            show(st.plotly_chart, pc.bar_fig(names, [rows[n]["avg_wait"] for n in names], "Average waiting time", "seconds"))
        with c2:
            show(st.plotly_chart, pc.bar_fig(names, [rows[n]["avg_queue"] for n in names], "Average queue length", "vehicles"))
        with c3:
            show(st.plotly_chart, pc.bar_fig(names, [rows[n]["throughput"] for n in names], "Throughput", "veh / hour"))
        after = st.selectbox("Before vs after: compare fixed-time with", ALGORITHMS, index=2)
        show(st.plotly_chart, pc.before_after_fig(rows, BASELINE, after))
        st.download_button("⬇ Download results as CSV", df.to_csv(index=False).encode("utf-8"),
                           "trafficTwin_results.csv", "text/csv")

with t_exp:
    f = DATA / "summary_mean_std.csv"
    summary_df = read_csv_safe(f)
    if summary_df.empty:
        st.info("Run `python -m experiments.run_experiments` to generate the multi-run study.")
    else:
        st.subheader("Mean and standard deviation over independent runs")
        show(st.dataframe, summary_df.round(3), hide_index=True)
        for name, title in [("improvement_vs_baseline.csv", "Improvement over fixed-time (%)"),
                            ("significance.csv", "Significance tests (Hybrid vs others)"),
                            ("ablation.csv", "Ablation raw runs"), ("popsize_study.csv", "Population size study raw runs"),
                            ("dimension_study.csv", "Dimension study raw runs")]:
            extra_df = read_csv_safe(DATA / name)
            if not extra_df.empty:
                with st.expander(title):
                    show(st.dataframe, extra_df.round(3), hide_index=True)
        if FIG.exists():
            for p in sorted(FIG.glob("*.png")):
                st.image(str(p), caption=p.stem)
            raw = DATA / "experiment_results.csv"
            if raw.exists():
                st.download_button("⬇ Download experiment results CSV", raw.read_bytes(),
                                   "experiment_results.csv", "text/csv")

with t_theory:
    st.subheader("Objective (minimised)")
    st.latex(r"J(x)=w_1\frac{W}{W_{ref}}+w_2\frac{\bar Q}{Q_{ref}}+w_3\frac{T}{T_{ref}}+w_4\frac{Q_{max}}{Q_{max,ref}}+P(x)")
    st.write("`ref` values come from the fixed-time baseline, so J = 1 means 'same as baseline' and J < 1 is better. "
             "P(x) penalises greens outside the bounds and cycles longer than the limit.")
    st.latex(r"q_j(t)=\max\big(0,\;q_j(t-1)+a_j(t)-c_j(t)\big)")
    st.subheader("Hybrid GA-PSO")
    st.markdown("""
1. **GA warm-up** explores the search space globally.
2. The best GA individuals are **transferred into the PSO swarm**.
3. **PSO** exploits them: `v = w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)`.
4. Every few iterations the personal bests act as **GA parents**; crossover + mutation offspring **replace the worst particles** (diversity).
5. Stops when the evaluation budget is used or progress stagnates.

All three optimisers receive the **same evaluation budget**, so the comparison is fair.
""")
