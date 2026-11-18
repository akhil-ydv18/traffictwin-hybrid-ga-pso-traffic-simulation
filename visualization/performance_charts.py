"""
performance_charts.py
=====================
* matplotlib functions (save_*) write PNG files for the report (docs/figures).
* plotly functions (*_fig) return interactive figures for the Streamlit dashboard.
"""
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:                                   # plotly is only needed by the dashboard
    import plotly.graph_objects as go
except ImportError:                    # pragma: no cover
    go = None

COLORS = {"Fixed-Time": "#6b7280", "GA": "#2563eb", "PSO": "#16a34a",
          "Hybrid GA-PSO": "#dc2626", "Hybrid (no injection)": "#f59e0b",
          "Hybrid (no warm-up)": "#8b5cf6"}
LABELS = {"avg_wait": "Average waiting time (s)", "avg_queue": "Average queue length (veh)",
          "max_queue": "Maximum queue length (veh)", "avg_travel": "Average travel time (s)",
          "throughput": "Throughput (veh/h)", "served": "Vehicles served", "objective": "Objective J"}


def _c(name):
    return COLORS.get(name, "#0ea5e9")


# =========================== matplotlib (PNG files) ==========================
def save_convergence(conv_df, scenario: str, path):
    """Mean +- std best-so-far J versus number of evaluations, per algorithm."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sub = conv_df[conv_df.scenario == scenario]
    for alg in ["GA", "PSO", "Hybrid GA-PSO"]:
        s = sub[sub.algorithm == alg].groupby("evals")["best_f"].agg(["mean", "std"]).fillna(0)
        if s.empty:
            continue
        ax.plot(s.index, s["mean"], label=alg, color=_c(alg), lw=2)
        ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=_c(alg), alpha=0.15)
    ax.set_xlabel("Objective evaluations"); ax.set_ylabel("Best objective J so far (lower = better)")
    ax.set_title(f"Convergence - {scenario} traffic (mean +- std over runs)")
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def save_metric_bars(summary, metric: str, path):
    """Grouped bars: scenarios on x axis, one bar per controller, error bar = std."""
    scns = list(dict.fromkeys(summary.scenario))
    algs = list(dict.fromkeys(summary.algorithm))
    w = 0.8 / len(algs)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for i, a in enumerate(algs):
        s = summary[summary.algorithm == a].set_index("scenario").loc[scns]
        ax.bar(np.arange(len(scns)) + i * w, s[f"{metric}_mean"], w, yerr=s[f"{metric}_std"],
               label=a, color=_c(a), capsize=2)
    ax.set_xticks(np.arange(len(scns)) + w * (len(algs) - 1) / 2); ax.set_xticklabels(scns)
    ax.set_ylabel(LABELS[metric]); ax.set_title(LABELS[metric] + " by controller")
    ax.grid(axis="y", alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def save_before_after(summary, scenario: str, algorithm: str, path):
    """Before (Fixed-Time) vs After (optimised) for several metrics of one scenario."""
    keys = ["avg_wait", "avg_queue", "max_queue", "avg_travel"]
    sub = summary[summary.scenario == scenario].set_index("algorithm")
    before = [sub.loc["Fixed-Time", f"{k}_mean"] for k in keys]
    after = [sub.loc[algorithm, f"{k}_mean"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - 0.2, before, 0.4, label="Before (Fixed-Time)", color=_c("Fixed-Time"))
    ax.bar(x + 0.2, after, 0.4, label=f"After ({algorithm})", color=_c(algorithm))
    ax.set_xticks(x); ax.set_xticklabels(["Wait (s)", "Avg queue", "Max queue", "Travel (s)"])
    ax.set_title(f"Before vs after optimisation - {scenario} traffic"); ax.grid(axis="y", alpha=0.3)
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def save_density(queue_series: Dict[str, np.ndarray], path, title="Traffic density over time"):
    """queue_series: controller name -> (T,4) queue array.  Plots total vehicles queued."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, q in queue_series.items():
        tot = q.sum(axis=1)
        k = 60                                           # 1-minute moving average for readability
        sm = np.convolve(tot, np.ones(k) / k, mode="valid")
        ax.plot(np.arange(len(sm)) + k // 2, sm, label=name, color=_c(name), lw=1.8)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Vehicles queued (all approaches)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=140); plt.close(fig)


def save_ablation(summary, scenario: str, path):
    sub = summary[summary.scenario == scenario]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(sub.algorithm, sub["objective_mean"], yerr=sub["objective_std"], capsize=3,
           color=[_c(a) for a in sub.algorithm])
    ax.set_ylabel("Test objective J (lower = better)")
    ax.set_title(f"Ablation - {scenario} traffic"); plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# =========================== plotly (dashboard) ===============================
def convergence_fig(results: Dict[str, "OptResult"], show_only=None):
    fig = go.Figure()
    for name, r in results.items():
        if show_only and name not in show_only:
            continue
        ev, best = zip(*r.eval_history)
        fig.add_trace(go.Scatter(x=list(ev), y=list(best), mode="lines", name=name,
                                 line=dict(color=_c(name), width=3)))
    fig.update_layout(title="Convergence (best objective J vs evaluations)",
                      xaxis_title="Objective evaluations", yaxis_title="Best J (lower = better)",
                      template="plotly_white", height=380, legend=dict(orientation="h", y=-0.25))
    return fig


def diversity_fig(results: Dict[str, "OptResult"]):
    fig = go.Figure()
    for name, r in results.items():
        d = r.extra.get("diversity")
        if d:
            fig.add_trace(go.Scatter(y=d, mode="lines", name=name, line=dict(color=_c(name))))
    fig.update_layout(title="Population diversity per generation / iteration",
                      xaxis_title="Generation / iteration", yaxis_title="Diversity (std / range)",
                      template="plotly_white", height=320, legend=dict(orientation="h", y=-0.3))
    return fig


def bar_fig(names: List[str], values: List[float], title: str, ylabel: str):
    fig = go.Figure(go.Bar(x=names, y=values, marker_color=[_c(n) for n in names],
                           text=[f"{v:.2f}" for v in values], textposition="outside"))
    fig.update_layout(title=title, yaxis_title=ylabel, template="plotly_white", height=340,
                      margin=dict(t=50, b=30))
    return fig


def before_after_fig(table_rows: Dict[str, Dict[str, float]], baseline="Fixed-Time", after=None):
    keys = ["avg_wait", "avg_queue", "max_queue", "avg_travel"]
    names = ["Wait (s)", "Avg queue", "Max queue", "Travel (s)"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before: " + baseline, x=names,
                         y=[table_rows[baseline][k] for k in keys], marker_color=_c(baseline)))
    fig.add_trace(go.Bar(name="After: " + after, x=names,
                         y=[table_rows[after][k] for k in keys], marker_color=_c(after)))
    fig.update_layout(barmode="group", title="Before vs after optimisation", template="plotly_white",
                      height=360, legend=dict(orientation="h", y=-0.2))
    return fig


def density_fig(queue_series: Dict[str, np.ndarray]):
    fig = go.Figure()
    for name, q in queue_series.items():
        tot = q.sum(axis=1)
        k = 30
        sm = np.convolve(tot, np.ones(k) / k, mode="valid")
        fig.add_trace(go.Scatter(x=np.arange(len(sm)) + k // 2, y=sm, mode="lines", name=name,
                                 line=dict(color=_c(name), width=2)))
    fig.update_layout(title="Traffic density over time (vehicles queued, 30 s moving average)",
                      xaxis_title="Time (s)", yaxis_title="Vehicles queued", template="plotly_white",
                      height=360, legend=dict(orientation="h", y=-0.25))
    return fig


def approach_queue_fig(queue_ts: np.ndarray, names=("N", "S", "E", "W")):
    fig = go.Figure()
    for j, n in enumerate(names):
        fig.add_trace(go.Scatter(y=queue_ts[:, j], mode="lines", name=n))
    fig.update_layout(title="Queue length per approach", xaxis_title="Time (s)",
                      yaxis_title="Vehicles", template="plotly_white", height=320,
                      legend=dict(orientation="h", y=-0.25))
    return fig
