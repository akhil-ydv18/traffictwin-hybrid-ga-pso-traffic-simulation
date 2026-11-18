"""
intersection.py
===============
Slow, object-oriented REFERENCE engine of a four-way intersection.

Every vehicle is a Python object stored in a FIFO queue (collections.deque).
It follows exactly the same rules as the fast vectorised engine in
traffic_simulator.py:

    1. vehicles arriving in second t join the queue of their approach;
    2. if the approach may discharge in second t (capacity = 1), the vehicle
       at the front of the queue leaves.

The reference engine is used by the unit tests to prove that the fast engine
is correct (both must give identical queue lengths), and for tiny demos.
"""
from collections import deque
from typing import Dict, List

import numpy as np

from .traffic_controller import APPROACHES
from .vehicle import Vehicle


class Intersection:
    def __init__(self):
        self.queues: Dict[str, deque] = {a: deque() for a in APPROACHES}
        self.vehicles: List[Vehicle] = []
        self._next_id = {a: 0 for a in APPROACHES}

    def step(self, t: int, arrivals_row, cap_row) -> List[int]:
        """Advance one second.  Returns the queue length of every approach."""
        lengths = []
        for j, a in enumerate(APPROACHES):
            for _ in range(int(arrivals_row[j])):             # 1. arrivals
                v = Vehicle(self._next_id[a], a, t)
                self._next_id[a] += 1
                self.queues[a].append(v)
                self.vehicles.append(v)
            if cap_row[j] and self.queues[a]:                  # 2. departure
                self.queues[a].popleft().departure_time = t
            lengths.append(len(self.queues[a]))
        return lengths


def run_object_engine(arrivals: np.ndarray, cap: np.ndarray):
    """Run the whole horizon with Vehicle objects.
    Returns (queue_ts [T,4], vehicles)."""
    horizon = arrivals.shape[0]
    inter = Intersection()
    q = np.zeros((horizon, 4), dtype=np.int64)
    for t in range(horizon):
        q[t] = inter.step(t, arrivals[t], cap[t])
    return q, inter.vehicles
