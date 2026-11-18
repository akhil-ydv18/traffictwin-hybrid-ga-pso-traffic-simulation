# Methodology

## 1. Problem
A four-way intersection serves two conflicting streams, North-South (NS) and East-West (EW). The signal
alternates between them. We choose the green time of each stream so that congestion is minimised.
There is no closed-form formula linking green times to congestion and traffic is random, so the problem is a
**noisy black-box optimisation** -- a natural fit for GA and PSO, which only need objective values.

## 2. Decision variables
The simulated period is split into K equal time blocks (default K = 3). In block k the controller uses
green times `gNS_k` and `gEW_k` (seconds):

    x = [gNS_1, gEW_1, gNS_2, gEW_2, ..., gNS_K, gEW_K]        (dimension 2K, default 6)

Yellow (3 s) and all-red clearance (1 s) are fixed safety constants and are **not** optimised.

Constraints
* box: `g_min <= g <= g_max` (10 s .. 60 s) -- enforced by clipping inside the algorithms;
* cycle: `C_k = gNS_k + gEW_k + 2(Y + R) <= C_max` (120 s) -- enforced by a penalty term.

## 3. Simulation model (1-second steps)
For approach j in {N, S, E, W}:

    a_j(t) ~ Poisson(lambda_j(t))                 random arrivals
    q_j(t) = max(0, q_j(t-1) + a_j(t) - c_j(t))   queue update
    c_j(t) = 1 if approach j may release a vehicle at second t, else 0

`c_j(t)` is 1 only during that stream's green, after a 2 s start-up lost time, once every 2 s (saturation flow
0.5 veh/s). The recursion above is solved for the whole time axis with vectorised NumPy (cumulative sum and running
minimum), so one 3600 s simulation takes about 0.6 ms. A slow object-oriented engine (`simulation/intersection.py`)
implements the same rules with real `Vehicle` objects; unit tests prove both give identical results.

Demand: each scenario gives vehicles per minute for N, S, E, W. A `shift` parameter makes the NS axis busier in the
first third of the run, balanced in the middle, and the EW axis busier in the last third. This is what makes
time-dependent plans (K > 1) worthwhile.

| Scenario | N, S (veh/min) | E, W (veh/min) | shift |
|---|---|---|---|
| Low | 3.0 | 2.5 | 30 % |
| Medium | 6.0 | 5.0 | 30 % |
| High | 8.5 | 7.0 | 30 % |
| Peak | 10.5 | 9.0 | 30 % |

## 4. Metrics
* **Average waiting time** = total vehicle-seconds spent in queues / vehicles that arrived (vehicles still queued
  at the end are charged their wait so far, so congestion cannot hide).
* **Average queue length** = mean queue per approach, averaged over time.
* **Maximum queue length** = longest queue on any approach at any time.
* **Average travel time** = 20 s free-flow time + average waiting time. (Because of this simple definition travel
  time is strongly correlated with waiting time -- a limitation of the model.)
* **Vehicles served / throughput** = vehicles that crossed the stop line, and per hour.

## 5. Objective (minimised)

    J(x) = w1*W/W_ref + w2*Q/Q_ref + w3*T/T_ref + w4*Qmax/Qmax_ref + P(x)

* `*_ref` = same metric of the **fixed-time baseline** (30 s / 30 s) on the same traffic. Dividing makes terms
  unit-less; **J = 1 means "same as baseline", J < 1 is better.**
* Default weights (0.35, 0.25, 0.15, 0.25), normalised to sum to 1; configurable.
* `P(x) = 0.05 * (sum of bound violations + sum of cycle-length excess)` in seconds.
* The algorithms **minimise J directly**. For plots where higher-is-better is more natural, `1/(1+J)` is used.

## 6. Fair experimental protocol
1. **Common random numbers** -- during optimisation every candidate is evaluated on the same fixed seeds, so
   differences in J come from the timings, not from lucky traffic.
2. **Train / test split** -- optimisation uses seeds 11, 12, 13; every number reported uses unseen seeds
   101-105. This prevents overfitting to one random traffic pattern.
3. **Equal evaluation budget** -- GA, PSO and Hybrid each get the same number of objective evaluations
   (default 1000), not just the same number of iterations.
4. **Independent runs** -- each configuration is repeated with different algorithm seeds (10 runs for the
   main study); mean and standard deviation are reported, plus a Mann-Whitney U test for Hybrid vs GA / PSO.
5. **Reproducibility** -- all randomness uses seeded NumPy generators; rerunning gives identical numbers
   (run times differ by machine).
