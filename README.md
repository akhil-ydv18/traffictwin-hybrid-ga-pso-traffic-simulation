# TrafficTwin - Hybrid GA-PSO Traffic Signal Optimization

M.Tech project for *Optimization Techniques for Intelligent Systems*.
A built-in intersection simulator + GA, PSO and a genuine hybrid GA-PSO (all from scratch) + Streamlit dashboard.
SUMO integration is listed as future scope (the built-in simulator works without it).

## 1. Setup (Windows + VS Code)
1. Install Python 3.10+ from python.org (tick **Add Python to PATH**).
2. VS Code: File > Open Folder > select `TrafficTwin`.
3. Open the terminal (Ctrl + `) and run:  `setup_windows.bat`
   (or manually: `python -m venv venv`, `venv\Scripts\activate`, `pip install -r requirements.txt`)
4. Ctrl+Shift+P > **Python: Select Interpreter** > choose the `venv` one.

macOS / Linux: `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

## 2. Run (always from the TrafficTwin folder, venv activated)
| Task | Command |
|---|---|
| Unit tests (23 tests, ~6 s) | `python -m unittest discover -s tests -v` |
| Dashboard | `streamlit run app.py` |
| Full experiments (~5 min) | `python -m experiments.run_experiments` |
| Quick experiment smoke test (~15 s) | `python -m experiments.run_experiments --quick` |
| Regenerate architecture / flowchart | `python -m visualization.diagrams` |

Dashboard: choose a scenario in the sidebar, press **Run Simulation** (fixed-time baseline) and
**Run Optimization** (GA, PSO, Hybrid really run, a few seconds). Tabs: Overview, Live intersection (animated),
Optimization (convergence + diversity), Comparison (tables, charts, CSV download), Saved experiments, Theory.

## 3. Project structure
```
app.py                     Streamlit dashboard
simulation/                vehicle.py, intersection.py (slow reference engine), traffic_controller.py, traffic_simulator.py (fast engine)
optimization/              fitness_function.py (objective J), genetic_algorithm.py, particle_swarm.py, hybrid_ga_pso.py, common.py
experiments/               evaluation.py, run_experiments.py
visualization/             traffic_visualization.py, performance_charts.py, diagrams.py
tests/                     unit tests + dashboard smoke test
data/                      CSV outputs written by run_experiments (generated, never hand-edited)
docs/                      methodology.md, algorithm_explanation.md, project_report.md, viva_questions.md,
                           results_summary.md (auto-generated), figures/*.png
```

## 4. Key design points
* Objective J is **minimised**; J = 1 is the fixed-time baseline, J < 1 is better.
* Fair comparison: equal evaluation budget, common random numbers, train seeds (11-13) vs unseen test seeds (101-105), 10 independent runs, mean +- std, Mann-Whitney test.
* The fast vectorised simulator is verified against a slow object-based engine by unit tests.
* Honest finding from the supplied run: all optimisers beat fixed-time (waiting time -31 % to -64 %), but the hybrid is statistically equal to GA / PSO on this problem (see `docs/project_report.md`).

## 5. Troubleshooting
* `ModuleNotFoundError: simulation` -> run commands from the `TrafficTwin` root folder.
* `streamlit` not found -> activate the venv or run `python -m streamlit run app.py`.
* Dashboard shows no experiment tables -> run `python -m experiments.run_experiments` once.
