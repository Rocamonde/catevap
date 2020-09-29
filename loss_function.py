from typing import List, Tuple

import numpy as np

from grid import Grid
from policies import Policy
from simulation import simulate, SimulationData
from vehicle import Vehicle


def loss_function(simulation_data: SimulationData) -> Tuple:
    """Calculates the loss function from the simulation data. This includes
    calling the other methods to calculate each of its terms.

    The functions calculate A, B, C are the mathematical implementation of the
    equations described in the paper SS2.3
    """
    vals = (calculate_A(simulation_data), calculate_B(simulation_data), *calculate_C(simulation_data))
    return vals


def _loss_function(val_A, val_B, val_C1, val_C2):
    """Actually calculates the loss function from the raw values"""
    return val_A ** 2 + val_B ** 2 + (1 - val_C1) ** 2 + val_C2 ** 2


def calculate_A(sd: SimulationData) -> float:
    avg_accels = []
    if len(sd.speed_histories) > 0:
        for speed in sd.speed_histories:
            speed_arr = np.array(speed)
            accels = np.diff(speed_arr)
            if len(accels) > 0:
                avg_accels.append(np.average(np.absolute(accels)) / Vehicle.max_deceleration)
            else:
                avg_accels.append(0)
        return np.average(np.array(avg_accels))
    return 0


def calculate_B(sd: SimulationData) -> float:
    try:
        return sd.collisions / sd.simtime
    except ZeroDivisionError:
        return 0


def calculate_C(sd: SimulationData) -> (float, float):
    avg_speeds = np.array(
        [np.average(np.array(speed)) if len(speed) > 0 else 0 for speed in sd.speed_histories ]) / Vehicle.max_speed
    return np.average(avg_speeds), np.std(avg_speeds)


def evaluate_policy(
        policy_cls: type(Policy),
        lattice: Grid,
        initial_speed: int,
        traffic_density: float,
        epochs: int
        ) -> [Tuple[Tuple, float]]:

    policy = policy_cls(initial_speed, traffic_density)
    data = simulate(
        epochs=epochs,
        initial_state=lattice,
        policy=policy
    )
    return loss_function(data)
