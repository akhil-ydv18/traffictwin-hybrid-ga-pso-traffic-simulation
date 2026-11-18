import unittest
import numpy as np
from simulation import (FixedTimeController, SimConfig, TrafficSimulator, build_demand,
                        scenario_demand, SCENARIOS)
from simulation.traffic_controller import TrafficController, NS_GREEN, EW_GREEN


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.cfg = SimConfig(duration=1200)
        self.sim = TrafficSimulator(self.cfg)

    def test_fast_engine_matches_object_engine(self):
        for name in SCENARIOS:
            arr = self.sim.generate_arrivals(scenario_demand(name, 1200), 5)
            c = FixedTimeController(25, 35)
            a, b = self.sim.simulate(c, arr), self.sim.simulate(c, arr, engine="object")
            self.assertTrue(np.array_equal(a.queue_ts, b.queue_ts))
            for k in a.metrics:
                self.assertAlmostEqual(a.metrics[k], b.metrics[k], places=9)

    def test_vehicle_conservation(self):
        arr = self.sim.generate_arrivals(scenario_demand("High", 1200), 3)
        r = self.sim.simulate(FixedTimeController(), arr)
        self.assertEqual(arr.sum(), r.dep_ts.sum() + r.queue_ts[-1].sum())

    def test_same_seed_same_traffic(self):
        d = scenario_demand("Medium", 1200)
        self.assertTrue(np.array_equal(self.sim.generate_arrivals(d, 9), self.sim.generate_arrivals(d, 9)))
        self.assertFalse(np.array_equal(self.sim.generate_arrivals(d, 9), self.sim.generate_arrivals(d, 10)))

    def test_only_green_approach_discharges(self):
        c = FixedTimeController(30, 30)
        cap, state = c.capacity(600)
        self.assertTrue(np.all(cap[state != NS_GREEN][:, :2] == 0))
        self.assertTrue(np.all(cap[state != EW_GREEN][:, 2:] == 0))
        self.assertGreater(cap[:, :2].sum(), 0)

    def test_cycle_length_fixed_time(self):
        c = FixedTimeController(30, 30)
        self.assertEqual(c.cycle_lengths()[0], 30 + 30 + 2 * (3 + 1))

    def test_more_demand_more_waiting(self):
        w = []
        for n in ("Low", "Peak"):
            arr = self.sim.generate_arrivals(scenario_demand(n, 1200), 1)
            w.append(self.sim.simulate(FixedTimeController(), arr).metrics["avg_wait"])
        self.assertLess(w[0], w[1])

    def test_zero_demand(self):
        arr = np.zeros((600, 4), dtype=np.int64)
        m = self.sim.simulate(FixedTimeController(), arr).metrics
        self.assertEqual(m["avg_wait"], 0.0)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            build_demand([1, 2, 3], 0.1, 100)
        with self.assertRaises(ValueError):
            build_demand([1, 1, 1, -1], 0.1, 100)
        with self.assertRaises(ValueError):
            SimConfig(duration=10)
        with self.assertRaises(ValueError):
            TrafficController([[np.nan, 3]])


if __name__ == "__main__":
    unittest.main()
