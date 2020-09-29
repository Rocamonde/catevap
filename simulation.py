from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from typing import TYPE_CHECKING

from grid import Grid, filter_dump_vehicles

if TYPE_CHECKING:
    from policies import Policy


class SimAlreadyRun(Exception):
    """Custom exception to prevent running a simulation object twice (this would tamper with the previous results).
    The way of doing it is to create a new simulation."""
    pass


class Simulation:
    """
    Simulation class that keeps the simulation grid (i.e. the lattice and the state, which are the same object,
    and is called just state or grid for convenience), the policy to be followed, as well as the necessary data to
    the analyse the performance of the policy.
    """
    def __init__(self, initial_state: Grid, policy: Policy, keep_history=False):
        """Initialises the Simulation. Providing an instantiated policy is required, as well as an initial state.
        From there, the simulation is able to generate subsequent states."""
        self.state: Grid = initial_state
        # The simulation data is all the counters required to feed to the loss function
        self.data: SimulationData = SimulationData()
        self.policy: Policy = policy
        # Flag to check whether the simulation has run yet
        self.__run: bool = False
        # Flag that determines whether we will store a history of the states
        self.keep_history: bool = keep_history

        # The simulation history is a backlog of all the previous states of the simulation, in the form of a list of
        # dictionaries, to be able to free the memory to unused vehicles if the full history is required.
        self.history: Optional[List[dict]] = list() if self.keep_history else None
        self.data_history: List[SimulationData] = list() if self.keep_history else None
        if self.keep_history:
            # Initial dump of the initial state
            self.update_history()

    def next_state(self):
        """Generates the next state of the simulation."""
        self.state = self.policy.next_state(self)
        if self.keep_history:
            self.update_history()

    def simulate(self, epochs, keep_alive=False):
        """
        Main method to call the simulation.

        This runs the simulation a given number of iterations, adds to the data vehicles still in the grid,
        and stores the number of iterations.
        """
        if self.__run:
            # Since we keep counters with the state of the simulation, we normally do not want to run the same
            # simulation object twice, as the data values would be conflated.
            raise SimAlreadyRun("Simulation has already been run. Please, create another simulation instance to "
                                "re-run the simulation.")

        for i in range(epochs):
            self.next_state()
            self.push_traffic_density()
            if self.keep_history:
                self.update_data_history()

        # We need to append the remaining objects that were not picked up
        # (i.e. vehicles that haven't left the grid yet)
        for x, y in self.state.indices():
            if (vehicle := self.state[x, y].vehicle) is not None:
                self.data.speed_histories.append(vehicle.state.speed_history)
                if not keep_alive:
                    self.state[x, y].vehicle = None
        self.data.simtime = epochs
        self.__run = True

    def update_history(self):
        self.history.append(self.state.dump())

    def push_traffic_density(self):
        self.data.traffic_density.append(self.get_traffic_density())

    def get_traffic_density(self):
        return len(self.state.vehicles) / self.state._grid.size

    def update_data_history(self):
        self.data_history.append(copy.deepcopy(self.data))


def simulate(initial_state, policy, epochs) -> SimulationData:
    """handy helper function that creates a simulation class automatically and runs the simulation """
    simulation = Simulation(initial_state, policy)
    simulation.simulate(epochs)
    return simulation.data


@dataclass
class SimulationData:
    """Dataclass to store all the simulation data"""
    collisions: float = 0
    speed_histories: List[Tuple] = field(default_factory=list)
    travelled_distance: int = 0
    simtime: int = 0
    traffic_density: List[int] = field(default_factory=list)
