# TrafficTwin: Hybrid GA-PSO Calibration of Traffic-Signal Timings

M.Tech (AI/ML) -- Optimization Techniques for Intelligent Systems

> All numbers in this report were produced by running `python -m experiments.run_experiments`
> (10 independent runs per cell, 1000 objective evaluations per run, unseen test traffic).
> The complete auto-generated tables are in `docs/results_summary.md` and raw data in `data/`.
> Re-running the script reproduces them exactly (run times vary with hardware).

## Abstract
Fixed-time traffic signals ignore actual demand and waste green time. This project builds a configurable
four-way intersection simulator and uses three metaheuristics -- a Genetic Algorithm (GA), Particle Swarm
Optimization (PSO) and a hybrid GA-PSO -- to optimise time-block-dependent signal plans by minimising a normalised,
weighted cost of waiting time, queue length, travel time and maximum queue. All algorithms are implemented from
scratch and compared with a fixed-time controller under four demand levels with equal evaluation budgets, common
random numbers and a train/test split of traffic seeds. All three optimisers reduce average waiting time by about
31-38 % (low, medium, high demand) and about 64 % (peak demand) compared with the 30 s/30 s fixed-time plan. The
hybrid GA-PSO performed on par with GA and PSO; no statistically significant difference was found (Mann-Whitney
p > 0.24 in every scenario). The report therefore documents the benefit of optimisation and honestly discusses why the
hybrid did not outperform its components on this problem.

## 1. Introduction
Urban congestion costs time and fuel. Signal timing is one of the cheapest levers, but the relation between timings and
delay is only available through simulation, is stochastic and non-convex. Metaheuristics need only objective values and
are therefore well suited.

## 2. Problem statement
Given stochastic arrival rates on four approaches, find green times (NS and EW) for each of K time blocks that minimise a
weighted congestion cost, subject to green-time bounds and a maximum cycle length.

## 3. Research objectives
1. Build a reproducible intersection simulator with low/medium/high/peak demand.
2. Implement GA, PSO and a genuine hybrid GA-PSO from scratch.
3. Compare them with a fixed-time controller fairly (equal budget, same seeds, unseen test traffic).
4. Study population size, dimension and ablation of the hybrid components.
5. Deliver an interactive dashboard for demonstration.

## 4. Literature review outline (references to verify before citing)
* Classical signal timing: Webster's delay-minimising cycle formula [1].
* Evolutionary computation: Holland's GA [2], Goldberg [3], Eiben and Smith [4].
* Swarm intelligence: Kennedy and Eberhart's PSO [5], inertia weight [6], convergence analysis [7], overview [8].
* Hybrid GA-PSO: early hybrids of GA and PSO [9].
* Traffic simulators: SUMO [10].
Each reference should be re-checked (title, venue, pages) in IEEE Xplore / Google Scholar before submission; I have
listed them from memory and could not verify them online while preparing this project.

## 5. Existing system
Fixed-time control: constant green times (here 30 s NS / 30 s EW, cycle 68 s). Simple, but cannot react to demand
changes. Webster-type optimal plans and actuated control are stronger baselines; adding them is future work.

## 6. Proposed methodology
See `docs/methodology.md` for full detail. Summary: simulate -> compute J -> optimiser proposes new plan, repeated within an
evaluation budget; report on unseen traffic.

### 6.1 Mathematical formulation
Decision vector `x = [gNS_1, gEW_1, ..., gNS_K, gEW_K]`, `10 <= g <= 60`, cycle `C_k = gNS_k + gEW_k + 8 <= 120`.

Queue dynamics: `q_j(t) = max(0, q_j(t-1) + a_j(t) - c_j(t))`, `a_j(t) ~ Poisson(lambda_j(t))`.

Objective (minimised):
`J(x) = w1 W/W_ref + w2 Q/Q_ref + w3 T/T_ref + w4 Qmax/Qmax_ref + P(x)`, weights (0.35, 0.25, 0.15, 0.25),
`P(x) = 0.05 * (sum of bound and cycle violations in seconds)`. `J = 1` is the baseline, `J < 1` is better.

### 6.2 GA
Real-coded; tournament selection (3), BLX-0.3 crossover (p = 0.9), Gaussian mutation (p = 0.25 per gene, sigma = 10 % of
range), elitism 2, population 20.

### 6.3 PSO
Swarm 20, `v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)`, w decreasing 0.9 -> 0.4, c1 = c2 = 1.8, velocity clamp 25 %
of range, clamped positions.

### 6.4 Hybrid GA-PSO architecture
GA warm-up (30 % of budget) -> transfer best 50 % of the population into the swarm (others random) -> PSO refinement ->
every 5 iterations GA crossover/mutation on personal bests replaces the worst 25 % of particles -> stop at budget.
Rationale: GA explores, PSO exploits, injection restores diversity.
Diagrams: `docs/figures/architecture.png` (system) and `docs/figures/hybrid_flowchart.png` (hybrid).

## 7. Experimental setup
* Duration 3600 s, K = 3 (6 variables), scenarios in `docs/methodology.md`.
* Train seeds 11, 12, 13; test seeds 101-105; budget 1000 evaluations per run for every optimiser.
* 10 independent runs (algorithm seeds 17, 7936, ...) for the main study and ablation; 8 runs for the size and dimension studies.
* Hardware for the reported run times: single CPU core in the development sandbox; one optimiser run took about 0.8-1.1 s.

## 8. Results and discussion
(Mean +- std over 10 runs, unseen test traffic. Full tables: `docs/results_summary.md`.)

| Scenario | Fixed-time wait (s) | GA | PSO | Hybrid GA-PSO |
|---|---|---|---|---|
| Low | 14.34 | 8.95 +- 0.32 | 8.94 +- 0.32 | 8.98 +- 0.19 |
| Medium | 16.54 | 11.55 +- 0.50 | 11.24 +- 0.33 | 11.37 +- 0.48 |
| High | 21.35 | 14.79 +- 0.52 | 14.63 +- 0.31 | 14.82 +- 0.52 |
| Peak | 56.98 | 20.63 +- 0.42 | 20.41 +- 0.45 | 20.61 +- 0.71 |

Average queue length falls by 30-38 % (low-high) and 64 % (peak); maximum queue falls by 14-25 % (low, medium), about 40 %
(high) and 63 % (peak). Objective improvement vs baseline: 24-30 % (low-high) and about 60 % (peak).

Findings
1. **Optimisation clearly helps.** The gain is largest at peak demand, where the fixed 30/30 plan cannot serve the busier
   axis during its heavy third of the run and queues build up.
2. **The hybrid did not beat GA or PSO.** Objective means differ in the third decimal place and no Mann-Whitney test is
   significant (p between 0.24 and 0.94). Interpretation: with 6 variables, a smooth-ish landscape and a common-random-number
   objective, all three methods reach nearly the same optimum within 1000 evaluations. The convergence plots
   (`docs/figures/convergence_*.png`, which show J on the *training* traffic) show PSO slightly ahead and the hybrid
   slightly behind in the early/middle phase, because the hybrid spends part of its budget on GA warm-up and injection.
3. **Throughput hardly changes (+0.1 %).** Almost every arriving vehicle is eventually served under all controllers in
   these scenarios, so throughput is not a discriminating metric here; waiting time and queues are.
4. **Ablation** (High and Peak, `data/ablation.csv`): GA-only, PSO-only, hybrid and the two reduced hybrids are all within about
   one standard deviation of each other except that "Hybrid (no warm-up)" had the lowest mean J in both scenarios
   (High 0.699 vs 0.721 for the full hybrid). This was not tested for significance and would need more runs; it suggests the
   GA warm-up may not pay for itself on this problem.
5. **Population size** (Peak): GA improved slightly with larger populations (J 0.407 -> 0.401 from 10 to 40), PSO was best at
   10 (0.399), the hybrid improved with size (0.412 -> 0.402). Differences are small relative to the run-to-run spread.
6. **Dimension** (Peak, same budget): K = 1 (2 variables) gives J about 0.59; K = 3 gives about 0.40, showing the value of
   time-dependent plans; K = 6 (12 variables) is slightly worse (about 0.42) because the budget is spread over a larger
   space. The hybrid still did not separate from GA/PSO.

## 9. Limitations and when the hybrid may not outperform
* The problem is low dimensional and well behaved, leaving little room for the hybrid's extra exploration; the hybrid
  also adds parameters (warm-up fraction, transfer fraction, injection period and fraction) that were not tuned.
* The baseline 30/30 plan is untuned; a Webster or actuated baseline would give smaller improvements.
* Single lane per approach, no turning movements, one intersection, synthetic demand profile, travel time defined as
  free-flow plus wait.
* Training and test traffic come from the same distribution; robustness to a different demand pattern was not tested.
* Results depend on the chosen weights; other weightings would change the optimum.
* Statistical power: 10 runs detect only fairly large differences.

## 10. Future scope
SUMO/TraCI integration for microscopic realism, multi-intersection coordination, turning movements and multiple lanes,
multi-objective optimisation (e.g. NSGA-II) instead of a weighted sum, harder benchmark problems (more intersections ->
higher dimension) where hybrids are more likely to help, tuned hybrid parameters, adaptive/online control and reinforcement learning.

## 11. Conclusion
A fully working, tested simulation-optimisation system was built. Optimised signal plans reduce waiting time by about 31-38 %
(low-high demand) and about 64 % (peak) versus fixed-time control. A genuine hybrid GA-PSO was implemented and evaluated
fairly, but on this 6-variable problem it was statistically equivalent to GA and PSO. The main contribution is therefore a
rigorous, reproducible comparison framework and an honest analysis of when hybridisation does and does not pay off.

## References (IEEE format; verify every entry before submission)
[1] F. V. Webster, "Traffic signal settings," Road Research Technical Paper No. 39, Her Majesty's Stationery Office, London, U.K., 1958.
[2] J. H. Holland, *Adaptation in Natural and Artificial Systems*. Ann Arbor, MI, USA: Univ. Michigan Press, 1975.
[3] D. E. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*. Reading, MA, USA: Addison-Wesley, 1989.
[4] A. E. Eiben and J. E. Smith, *Introduction to Evolutionary Computing*. Berlin, Germany: Springer, 2003.
[5] J. Kennedy and R. Eberhart, "Particle swarm optimization," in *Proc. IEEE Int. Conf. Neural Networks*, Perth, WA, Australia, 1995, vol. 4, pp. 1942-1948.
[6] Y. Shi and R. Eberhart, "A modified particle swarm optimizer," in *Proc. IEEE Int. Conf. Evol. Comput.*, Anchorage, AK, USA, 1998, pp. 69-73.
[7] M. Clerc and J. Kennedy, "The particle swarm - explosion, stability, and convergence in a multidimensional complex space," *IEEE Trans. Evol. Comput.*, vol. 6, no. 1, pp. 58-73, Feb. 2002.
[8] R. Poli, J. Kennedy, and T. Blackwell, "Particle swarm optimization: An overview," *Swarm Intell.*, vol. 1, no. 1, pp. 33-57, 2007.
[9] J. Robinson, S. Sinton, and Y. Rahmat-Samii, "Particle swarm, genetic algorithm, and their hybrids: Optimization of a profiled corrugated horn antenna," in *Proc. IEEE Antennas Propag. Soc. Int. Symp.*, 2002, vol. 1, pp. 314-317.
[10] P. A. Lopez *et al.*, "Microscopic traffic simulation using SUMO," in *Proc. IEEE Int. Conf. Intell. Transp. Syst. (ITSC)*, Maui, HI, USA, 2018, pp. 2575-2582.
