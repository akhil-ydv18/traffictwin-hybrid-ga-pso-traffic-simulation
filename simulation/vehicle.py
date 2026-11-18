"""
vehicle.py
==========
A tiny Vehicle record plus a helper that rebuilds individual vehicles
from aggregate arrival/departure counts.

Why do we need individual vehicles?
-----------------------------------
The fast simulator works on *counts per second* (very quick, needed because
the optimizers call the simulator thousands of times).  Individual Vehicle
objects are used for (a) the slow reference engine that validates the fast
one, and (b) detailed analysis such as the waiting-time histogram.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Vehicle:
    vid: int                      # id inside its approach
    approach: str                 # "N", "S", "E" or "W"
    arrival_time: int             # second at which it joined the queue
    departure_time: Optional[int] = None   # second it crossed the stop line

    @property
    def served(self) -> bool:
        return self.departure_time is not None

    def waiting_time(self, horizon: int) -> int:
        """Seconds spent in the queue.  A vehicle that has not left by the end
        of the simulation is charged the time waited so far (horizon - arrival),
        otherwise unserved vehicles would silently hide congestion."""
        end = self.departure_time if self.served else horizon
        return int(end - self.arrival_time)


def vehicles_from_counts(arrivals_col: np.ndarray, departures_col: np.ndarray,
                         approach: str, horizon: int) -> List[Vehicle]:
    """Rebuild Vehicle objects of ONE approach using FIFO matching:
    the k-th vehicle to arrive is the k-th vehicle to depart."""
    t = np.arange(horizon)
    arr_times = np.repeat(t, arrivals_col.astype(int))
    dep_times = np.repeat(t, departures_col.astype(int))
    vehicles = []
    for i, at in enumerate(arr_times):
        dt = int(dep_times[i]) if i < len(dep_times) else None
        vehicles.append(Vehicle(i, approach, int(at), dt))
    return vehicles
