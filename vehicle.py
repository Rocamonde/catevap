from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, List

import numpy as np

from grid import Cell


@dataclass
class Vehicle:
    """Simple dataclass to store the properties and state of the vehicle"""
    state: Vehicle.State
    props: Vehicle.Props

    @dataclass
    class Position:
        """General dataclass to describe positions in the grid."""
        x: int
        y: int

        @classmethod
        def from_array(cls, arr: Sequence) -> Vehicle.Position:
            """Loads Position object from an array"""
            return cls(x=arr[0], y=arr[1])

        def to_array(self) -> np.ndarray:
            """Dumps position object to an array"""
            return np.array([self.x, self.y])

    @dataclass
    class State:
        """
        Dataclass to keep the state of the vehicle.
        It stores its direction, speed, history of past speeds and travelled distance (both for analsysis purposes),
        a next_speed attribute to avoid conflicting values when calculating the next value of speed, a collided flag
        that indicates if the vehicle should still be kept halted (recent collision), and rtc,  the remaining collision
        time penalty.
        """
        direction: Optional[Cell.Direction]
        speed: int
        speed_history: List[int] = field(default_factory=list)
        travelled_distance: int = 0
        next_speed: Optional[int] = None
        collided: bool = False
        rct: int = 0

        @property
        def velocity(self) -> np.ndarray:
            return self.speed * Cell.Direction.to_vector(self.direction)

        @property
        def uvel(self) -> np.ndarray:
            if self.velocity[0] == 0.0 and self.velocity[1] == 0.0:
                return self.velocity
            return self.velocity / np.linalg.norm(self.velocity)

    @dataclass
    class Props:
        max_speed: int = 15
        max_acceleration: int = 3
        max_deceleration: int = 6

    def dump(self) -> dict:
        return {
            'speed': self.state.speed,
            'direction': self.state.direction,
            'max_speed': self.props.max_speed,
            'collided': self.state.collided,
            'rct': self.state.rct,
            'max_acceleration': self.props.max_acceleration,
            'max_deceleration': self.props.max_deceleration,
        }

    def update_speed(self):
        if (next_speed := self.state.next_speed) is not None:
            self.state.speed = next_speed
            self.state.next_speed = None

    # Dirty trick for this particular simulation (it can be modified to allow maximum values).
    max_speed = 15
    max_acceleration = 3
    max_deceleration = 6


# Type Variable to store a tuple of coordinates and the vehicle object
LocatedVehicle = Tuple[Tuple[int, int], Vehicle]
