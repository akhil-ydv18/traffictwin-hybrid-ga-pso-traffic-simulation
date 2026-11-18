"""
diagrams.py - draws the system-architecture diagram and the hybrid GA-PSO
flowchart with matplotlib (no extra tools needed).

    python -m visualization.diagrams        ->  docs/figures/architecture.png, hybrid_flowchart.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


def _box(ax, x, y, w, h, text, color="#dbeafe", edge="#1e3a8a", fs=9):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=color, ec=edge, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#374151"))
    if text:
        ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, text, fontsize=8, color="#374151", va="center")


def architecture(path):
    fig, ax = plt.subplots(figsize=(10, 6)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 1.6, 5.1, 2.6, 0.9, "Traffic scenario\n(arrival rates, duration,\nLow / Medium / High / Peak)", "#fef9c3", "#854d0e")
    _box(ax, 5.0, 5.1, 2.6, 0.9, "Streamlit dashboard (app.py)\nparameters, buttons, charts", "#fce7f3", "#9d174d")
    _box(ax, 8.4, 5.1, 2.6, 0.9, "Experiments\nrun_experiments.py\n(multi-run, ablation)", "#fce7f3", "#9d174d")
    _box(ax, 5.0, 3.6, 3.4, 1.0, "Optimizer\nGA  |  PSO  |  Hybrid GA-PSO", "#fee2e2", "#991b1b", 10)
    _box(ax, 5.0, 1.9, 3.4, 1.0, "Objective J(x)\nnormalised weighted cost + penalties", "#dcfce7", "#166534", 10)
    _box(ax, 1.6, 1.9, 2.6, 1.0, "Signal controller\nfixed-time / plan (K blocks)", "#dbeafe", "#1e3a8a")
    _box(ax, 8.4, 1.9, 2.6, 1.0, "Traffic simulator\n4 approaches, queues,\nPoisson arrivals", "#dbeafe", "#1e3a8a")
    _box(ax, 5.0, 0.4, 6.5, 0.6, "Results: metrics, CSV files, convergence & comparison charts", "#f3f4f6", "#374151")
    _arrow(ax, 2.9, 4.7, 4.0, 3.9); _arrow(ax, 5.0, 4.65, 5.0, 4.1); _arrow(ax, 7.1, 4.7, 6.0, 3.9)
    _arrow(ax, 5.0, 3.1, 5.0, 2.4, "candidate plan x"); _arrow(ax, 6.7, 1.9, 7.1, 1.9)
    _arrow(ax, 3.3, 1.9, 2.9, 1.9); _arrow(ax, 1.6, 1.4, 4.2, 0.65); _arrow(ax, 5.0, 1.4, 5.0, 0.7)
    ax.text(5.4, 2.75, "J(x)", fontsize=9, color="#991b1b")
    ax.set_title("TrafficTwin - system architecture", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def hybrid_flowchart(path):
    fig, ax = plt.subplots(figsize=(8, 10)); ax.set_xlim(0, 8); ax.set_ylim(0, 10.5); ax.axis("off")
    steps = [(9.9, "START: set budget, population size, parameters", "#f3f4f6"),
             (8.8, "PHASE 1 - GA warm-up\nrandom population -> evaluate -> selection,\ncrossover, mutation, elitism", "#dbeafe"),
             (7.4, "PHASE 2 - transfer\ntop GA individuals become particles\n(+ random newcomers for diversity)", "#fef9c3"),
             (6.0, "PHASE 3 - PSO iteration\nupdate velocity & position, clamp to bounds,\nevaluate, update pbest and gbest", "#dcfce7"),
             (4.6, "Every P iterations?", "#fde68a"),
             (3.3, "PHASE 4 - GA injection\npbest as parents -> crossover + mutation ->\noffspring replace the worst particles", "#dbeafe"),
             (1.9, "Budget used up or no improvement for `patience` iterations?", "#fde68a"),
             (0.7, "RETURN gbest (best signal plan) + convergence history", "#f3f4f6")]
    for y, t, c in steps:
        _box(ax, 4, y, 5.6, 0.95, t, c, fs=9)
    for a, b in zip(steps[:-1], steps[1:]):
        if b[0] == 3.3:
            continue
        _arrow(ax, 4, a[0] - 0.5, 4, b[0] + 0.5)
    _arrow(ax, 4, 4.1, 4, 3.8, "yes"); _arrow(ax, 4, 2.8, 4, 2.4)
    ax.text(4.1, 1.3, "yes: stop", fontsize=8)
    ax.annotate("", xy=(6.8, 4.6), xytext=(6.8, 4.6))
    ax.text(6.9, 4.9, "no (skip)", fontsize=8)
    ax.plot([6.8, 7.5, 7.5, 6.8], [4.6, 4.6, 1.9, 1.9], color="#374151", lw=1.5)   # skip injection
    ax.plot([6.8, 7.8, 7.8, 6.8], [1.9, 1.9, 6.0, 6.0], color="#374151", lw=1.5, ls="--")
    ax.annotate("", xy=(6.8, 6.0), xytext=(7.0, 6.0), arrowprops=dict(arrowstyle="-|>", color="#374151"))
    ax.text(7.95, 3.5, "no: next iteration", fontsize=8, rotation=90)
    ax.set_title("Hybrid GA-PSO flowchart", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    out = ROOT / "docs" / "figures"; out.mkdir(parents=True, exist_ok=True)
    architecture(out / "architecture.png"); hybrid_flowchart(out / "hybrid_flowchart.png")
    print("diagrams written to", out)
