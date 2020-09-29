from __future__ import annotations

from enum import Enum
from typing import Optional, Iterable, Set, TYPE_CHECKING, Union, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from vehicle import Vehicle, LocatedVehicle
    VehicleList = List[LocatedVehicle]


class Grid:
    """
    A class to store the lattice arrangement and the state of the lattice cells.
    Under the hood, it keeps a numpy array and exposes handy methods to perform simple operations on the array more
    quickly.
    """
    def __init__(self, shape):
        """Creates an empty grid with the provided shape and with array slots of type Cell"""
        self._grid = np.empty(shape, dtype=Cell)

    def get(self, position: Vehicle.Position) -> Cell:
        return self[position.x, position.y]

    def __setitem__(self, *args, **kwargs):
        """Exposes __setitem__ from numpy.ndarray"""
        return self._grid.__setitem__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs) -> Union[Cell, np.ndarray]:
        """Exposes __getitem__ from numpy.ndarray"""
        return self._grid.__getitem__(*args, **kwargs)

    @classmethod
    def new_from(cls, grid: Grid) -> Grid:
        """
        Creates a new grid from an old grid.
        It creates new cells with equal properties, but without referencing the vehicles.
        :param grid: old grid
        :return: new grid
        """
        new = cls(grid._grid.shape)
        for x, y in new.indices():
            new[x, y] = Cell.new_from(grid[x, y])
        return new

    def indices(self):
        """Syntactic sugar for getting all the pairs of the indices of the array"""
        return np.ndindex(self._grid.shape)

    def puts_vehicle(self, position: Vehicle.Position, vehicle: Vehicle):
        """Helper to put a vehicle in a specific location"""
        try:
            # We might want to put a vehicle in a location that is not within the array.
            # If that is the case, we let the vehicle be unassigned and garbage-collected by Python.
            self.get(position).vehicle = vehicle
        except IndexError:
            pass

    def dump(self):
        """Dumps the current grid to a plain Python object that can be serialised (i.e. dicts and lists)"""
        new = np.empty(self._grid.shape, dtype=dict)
        for x, y in self.indices():
            new[x, y] = self[x, y].dump()
        return new

    @classmethod
    def new_from_arr(cls, arr) -> Grid:
        """Creates a Grid wrappe for an existing array"""
        new = cls(arr.shape)
        for x, y in new.indices():
            new[x, y] = Cell.new_from_dict(arr[x, y])
        return new

    @property
    def vehicles(self) -> VehicleList:
        """Gets all the vehicles for the current grid"""
        return self.get_vehicles(self._grid)

    @staticmethod
    def get_vehicles(arr: np.ndarray) -> VehicleList:
        """Static method to get all the vehicles for a given array"""
        vehicles: VehicleList = []
        for x, y in np.ndindex(arr.shape):
            if (vehicle := arr[x, y].vehicle) is not None:
                vehicles.append((
                    (x, y), vehicle
                ))
        return vehicles


class Cell:
    """Custom Python object that holds the properties of the cell, as well as a reference to the vehicle, if any (i.e.
    the cell  state)
    Cells have a type, a set of allowed directions, and possibly a vehicle.
    """
    def __init__(self, cell_type: Optional[Cell.Type] = None,
                 allowed_directions: Optional[Iterable[Cell.Direction]] = None,
                 vehicle: Optional[Vehicle] = None) -> None:

        self.cell_type: Cell.Type = cell_type or Cell.Type.block
        self.directions: Set[Cell.Direction]
        self.vehicle: Optional[Vehicle]

        if self.cell_type == Cell.Type.block:
            self.directions = set()
            self.vehicle = None
        else:
            try:
                self.directions = set(allowed_directions)
            except TypeError:
                self.directions = set()
            self.vehicle = vehicle

    @property
    def is_multi_direction(self) -> bool:
        return len(self.directions) > 1

    @property
    def has_vehicle(self) -> bool:
        return self.vehicle is not None

    @property
    def can_have_vehicle(self) -> bool:
        return self.cell_type == Cell.Type.road

    class Type(Enum):
        """Simple Enum class to store the types of cells available"""
        road = 'road'
        block = 'block'

    class Direction(Enum):
        """Simple Enum class to store the possible 2D directions"""
        north = 'north'
        west = 'west'
        east = 'east'
        south = 'south'

        @classmethod
        def to_vector(cls, d: Cell.Direction) -> np.ndarray:
            """Converts a direction into a unit vector, i.e. a numpy array"""
            if d == cls.north:
                return np.array([0, 1])
            if d == cls.south:
                return np.array([0, -1])
            if d == cls.west:
                return np.array([-1, 0])
            if d == cls.east:
                return np.array([1, 0])

    @classmethod
    def new_from(cls, grid_cell: Cell):
        """Creates a new cell from an old one, without the vehicle, but all the other properties the same"""
        return cls(cell_type=grid_cell.cell_type,
                   allowed_directions=grid_cell.directions)

    def dump(self):
        """Dumps a cell to a dictionary, storing all its properties and attributes"""
        return {
            'cell_type': self.cell_type,
            'directions': self.directions,
            'vehicle': self.vehicle.dump() if self.vehicle else None,
            'is_multi_direction': self.is_multi_direction,
            'has_vehicle': self.has_vehicle,
            'can_have_vehicle': self.can_have_vehicle
        }

    @classmethod
    def new_from_dict(cls, d: dict) -> Cell:
        """Creates a new cell from a dictionary (i.e. loads the dict)"""
        if (cell_type := d.get('cell_type')) is not None:
            d['cell_type'] = Cell.Type(cell_type)
        else:
            raise ValueError('All cells must specify a cell type.')

        if (dirs := d.get('allowed_directions')) is not None:
            d['allowed_directions'] = [Cell.Direction(dir) for dir in dirs]

        return cls(**d)


def filter_dump_vehicles(dump):
    """Filters out from a grid dump all the existing vehicles"""
    vehicles = []
    for x, y in np.ndindex(dump.shape):
        if (vehicle := dump[x, y]['vehicle']) is not None:
            vehicles.append(((x, y), vehicle['speed'], vehicle['collided']))
    return vehicles
