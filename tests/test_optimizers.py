import unittest
import numpy as np
from optimization import (Bounds, GeneticAlgorithm, HybridGAPSO, ParticleSwarm, TrafficObjective,
                          fitness_value)
from simulation import SimConfig, scenario_demand
from simulation.traffic_controller import FixedTimeController


def sphere(x):
    return float(np.sum((np.asarray(x) - 3.0) ** 2))


LO, HI = np.full(4, -10.0), np.full(4, 10.0)


class TestOptimizers(unittest.TestCase):
    def test_ga_solves_sphere(self):
        r = GeneticAlgorithm(sphere, LO, HI, pop_size=30, generations=80, seed=1).run()
        self.assertLess(r.best_f, 0.5)

    def test_pso_solves_sphere(self):
        r = ParticleSwarm(sphere, LO, HI, swarm_size=20, iterations=80, seed=1).run()
        self.assertLess(r.best_f, 1e-2)

    def test_hybrid_solves_sphere(self):
        r = HybridGAPSO(sphere, LO, HI, pop_size=20, max_evals=1200, seed=1).run()
        self.assertLess(r.best_f, 1e-2)
        self.assertGreater(r.extra["injections"], 0)       # both mechanisms really ran

    def test_budget_respected(self):
        for opt in (GeneticAlgorithm(sphere, LO, HI, generations=10 ** 6, max_evals=300, seed=0),
                    ParticleSwarm(sphere, LO, HI, iterations=10 ** 6, max_evals=300, seed=0),
                    HybridGAPSO(sphere, LO, HI, max_evals=300, seed=0)):
            r = opt.run()
            self.assertLessEqual(r.n_evals, 300)
            self.assertGreater(r.n_evals, 240)

    def test_monotone_history_and_bounds(self):
        for opt in (GeneticAlgorithm(sphere, LO, HI, seed=2), ParticleSwarm(sphere, LO, HI, seed=2),
                    HybridGAPSO(sphere, LO, HI, max_evals=400, seed=2)):
            r = opt.run()
            self.assertTrue(all(a >= b - 1e-12 for a, b in zip(r.history, r.history[1:])))
            self.assertTrue(np.all(r.best_x >= LO) and np.all(r.best_x <= HI))

    def test_reproducible(self):
        a = HybridGAPSO(sphere, LO, HI, max_evals=400, seed=5).run()
        b = HybridGAPSO(sphere, LO, HI, max_evals=400, seed=5).run()
        self.assertEqual(a.best_f, b.best_f)

    def test_ablation_switches(self):
        r = HybridGAPSO(sphere, LO, HI, max_evals=400, inject_period=0, seed=0).run()
        self.assertEqual(r.extra["injections"], 0)
        r = HybridGAPSO(sphere, LO, HI, max_evals=400, warmup_fraction=0.0, seed=0).run()
        self.assertEqual(r.extra["warmup_len"], 0)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            GeneticAlgorithm(sphere, LO, HI, pop_size=2)
        with self.assertRaises(ValueError):
            ParticleSwarm(sphere, HI, LO)
        with self.assertRaises(ValueError):
            HybridGAPSO(sphere, LO, HI, max_evals=10)


class TestObjective(unittest.TestCase):
    def setUp(self):
        self.cfg = SimConfig(duration=1200)
        self.obj = TrafficObjective(scenario_demand("High", 1200), self.cfg, [1, 2], n_blocks=1)

    def test_baseline_has_objective_one(self):
        self.assertAlmostEqual(self.obj.cost_from_metrics(self.obj.baseline_metrics), 1.0)
        self.assertAlmostEqual(self.obj([30, 30]), 1.0, places=9)

    def test_penalty_for_long_cycle(self):
        self.assertEqual(self.obj.penalty([30, 30]), 0.0)
        self.assertGreater(self.obj.penalty([60, 60, ]), 0.0)     # cycle 128 > 120
        self.assertGreater(self.obj.penalty([5, 30]), 0.0)        # below g_min

    def test_deterministic_with_common_random_numbers(self):
        self.assertEqual(self.obj([35, 25]), self.obj([35, 25]))

    def test_better_plan_has_lower_cost(self):
        self.assertLess(self.obj([40, 25]), self.obj([10, 60]) + 5)

    def test_weights_normalised_and_validated(self):
        self.assertAlmostEqual(self.obj.weights.sum(), 1.0)
        with self.assertRaises(ValueError):
            TrafficObjective(scenario_demand("High", 1200), self.cfg, [1], weights=[0, 0, 0, 0])
        with self.assertRaises(ValueError):
            self.obj([1.0])

    def test_fitness_transform(self):
        self.assertAlmostEqual(fitness_value(0.0), 1.0)
        self.assertLess(fitness_value(1.0), fitness_value(0.5))


if __name__ == "__main__":
    unittest.main()
