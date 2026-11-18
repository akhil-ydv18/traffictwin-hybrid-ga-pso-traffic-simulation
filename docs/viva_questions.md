# Viva questions and answers

## A. Problem and modelling
**Q1. Why is signal timing an optimisation problem?**
Green times trade off the NS and EW queues and the cycle length trades delay against lost time. We want the plan with
the lowest congestion cost for the given demand, subject to limits on green and cycle length.

**Q2. Why metaheuristics instead of gradient descent or grid search?**
The objective comes from a stochastic simulation: no analytic gradient, noisy, non-convex, plateaus because green
times are rounded to seconds. GA and PSO need only objective values. Grid search is possible for 2 variables but
explodes with dimension (with 6 variables and 51 values each that is 51^6 = 1.8e10 simulations).

**Q3. What are the decision variables and constraints?**
2K green times (NS and EW for each of K blocks), 10-60 s each (box constraint, clipped), cycle <= 120 s (penalty).
Yellow and all-red are fixed safety constants.

**Q4. How does your simulator work, and how do you know it is correct?**
Per second: Poisson arrivals join the queue, and an approach releases one vehicle if its signal permits
(green, after start-up lost time, every 2 s). The fast engine solves `q(t) = max(0, q(t-1) + a(t) - c(t))` with
NumPy; a slow object-based engine with real Vehicle objects implements the same rules, and a unit test shows
identical queues and metrics. Another test checks vehicle conservation (arrivals = departures + remaining queue).

**Q5. Why does K > 1 (time blocks) matter?**
Demand shifts during the run (NS-heavy, balanced, EW-heavy). A single fixed plan cannot suit all three periods,
and K = 1 reduces the problem to 2 variables that any method solves easily.

**Q6. Limitations of the model?**
Single lane, no turning movements, no pedestrians or vehicle dynamics, one intersection (no coordination), demand
shift is synthetic, travel time = free-flow + wait so it is correlated with waiting time. SUMO integration is future
work.

## B. Objective function
**Q7. Write the objective and explain normalisation.**
`J = w1 W/Wref + w2 Q/Qref + w3 T/Tref + w4 Qmax/Qmaxref + P`. Dividing by the fixed-time baseline makes the terms
unit-less; J = 1 = baseline, J < 1 = better.

**Q8. Minimise or maximise?**
Minimise J directly. A higher-is-better "fitness" 1/(1+J) is used only for display.

**Q9. Why a penalty and not a hard constraint?**
Bounds are guaranteed by clipping. The cycle limit couples variables, so it is handled with a penalty proportional to
the violation (0.05 per second), which is zero for feasible plans.

**Q10. What are common random numbers and why a train/test split?**
Every candidate is evaluated on the same fixed traffic seeds, so comparisons reflect timings not luck. Optimisation
uses seeds 11-13; reported results use unseen seeds 101-105 so we measure generalisation, not memorisation.

## C. Algorithms
**Q11. Explain GA operators you used.**
Tournament selection (size 3), BLX-0.3 crossover, per-gene Gaussian mutation, elitism of 2.

**Q12. Explain the PSO update and the roles of w, c1, c2.**
`v = w v + c1 r1 (pbest - x) + c2 r2 (gbest - x); x = x + v`. w = inertia (explore vs exploit, decreased 0.9 -> 0.4),
c1 = pull to own best, c2 = pull to swarm best.

**Q13. Why hybridise GA and PSO? What is genuinely hybrid in your code?**
GA explores, PSO exploits. In one run GA individuals are transferred into the swarm and PSO personal bests become
GA parents whose offspring replace the worst particles. Without both mechanisms it would just be GA or PSO.

**Q14. What would happen with no GA injection or no warm-up?**
That is the ablation (`Hybrid (no injection)`, `Hybrid (no warm-up)`); see the ablation table in `docs/results_summary.md`
for the measured effect.

**Q15. How did you make the comparison fair?**
Equal evaluation budget (not equal iterations), same traffic scenarios and seeds, several independent runs with
fixed seeds, mean +- std, and a Mann-Whitney test.

**Q16. Does the hybrid always win?**
No. When the landscape is easy (low traffic, few variables) many plans are near-optimal and GA, PSO and the hybrid
tie; the hybrid also adds extra parameters to tune. Always check the measured tables rather than assuming.

**Q17. What is premature convergence?**
The swarm collapses to one point before the global optimum is found. Diversity (std of positions / range) drops. The
GA injection step is designed to counter this; the diversity chart in the dashboard shows it.

**Q18. How are boundaries handled?**
GA: children are clipped to bounds. PSO: positions clamped and the velocity component that hit the wall set to
zero; velocity is limited to 25 % of the range.

**Q19. What stopping criteria are used?**
Evaluation budget (primary, for fairness). Optional stagnation stop after `patience` iterations without improvement.

**Q20. Time complexity?**
One evaluation costs O(S * T) for S seeds and T seconds (vectorised). GA/PSO/Hybrid cost O(B * S * T) for budget B.

## D. Results and interpretation
**Q21. How do you report the results?**
Mean +- std over independent runs on unseen traffic, percentage improvement over fixed-time, significance test.
See `docs/results_summary.md` (auto-generated from the actual run).

**Q22. Why do GA, PSO and Hybrid give similar results on a 6-variable problem?**
The landscape is relatively smooth and low-dimensional, so all three reach near the same optimum within the budget.
Differences show mostly in convergence speed and in harder (higher-dimensional) settings -- see the dimension study.

**Q23. How could the project be extended?**
SUMO / TraCI integration, multi-intersection coordination, turning movements, multi-objective optimisation (NSGA-II),
adaptive (real-time) control, reinforcement learning.

**Q24. How is reproducibility ensured?**
All random generators are seeded; the experiment script is deterministic; results are written to CSV by code, never typed by hand.

**Q25. What did you validate with tests?**
Simulator equivalence and conservation, constraint penalty, baseline J = 1, optimisers on a known function
(sphere), budget limits, monotone convergence history, bounds, reproducibility, ablation switches, input validation.
