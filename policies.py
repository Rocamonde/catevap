from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, TYPE_CHECKING, Union

import numpy as np

from grid import Grid, Cell
from vehicle import Vehicle, LocatedVehicle

if TYPE_CHECKING:
    from simulation import Simulation

# We define a Type Variable to conveniently indicate the return type of the
# `Policy.detect_collision` method.
Collision = Tuple[Vehicle.Position, LocatedVehicle]


class Policy(ABC):
    """
    Base policy class. Here, we define general properties of our policies, like how vehicles are moved across the
    lattice, and what to do when two vehicle crash. Particular policies subclass this Abstract Base Class (ABC) to
    implement their own `update_velocity` method, that will define how the velocities of the vehicles evolve (i.e.
    when they accelerate or decelerate).

    The primary method of Policy is `Policy.next_state`. This takes in a simulation object (see simulation.py) and
    returns the new Grid object to assign to the simulation. The old Grid object is NOT modified in place,
    but the corresponding vehicles are.

    Apart from `next_state`, the other main method is `update_position`. This describes how, for a given simulation
    with its corresponding state, the positions of the vehicles will be updated. Other methods are helper methods
    that implement different mathematical or computational components of the base policy, as described in the paper.
    """
    def __init__(self, initial_speed: int, traffic_density: float):
        """
        Initialises the base Policy object.
        :param initial_speed: initial speed for all the vehicles generated in sources (see `.update_sources`).
        :param traffic_density: frequency of vehicles being added to a given source cell (i.e. probability)
        """
        self.initial_speed: int = initial_speed
        self.traffic_density: float = traffic_density

    def next_state(self, simulation: Simulation) -> Grid:
        """
        Generates the next state for a given simulation object. First, the velocities are updated according to the
        implementation of this ABC. Second, the positions are updated, according to `self.update_position`. Last,
        new vehicles are added if it corresponds. the new state is formed (a copy of the lattice is created with the
        new arrangements).

        The lattice is copied, because while re-arranging the vehicles (i.e. iterating over the new state),
        the old positions of vehicles are required, which is not possible if the array is modified on the fly.

        :param simulation: Simulation object to perform the policy operation on
        :return: new Grid object that represents the new lattice and its state
        """
        prev_state = simulation.state
        # This will update the `.speed` attribute in the vehicles.
        self.update_velocity(prev_state)
        # This will return a new state with the vehicles in the updated positions.
        new_state = self.update_position(simulation)
        # This will add vehicles where it corresponds, on top of the
        # newly-generated state.
        self.update_sources(new_state)
        return new_state

    @abstractmethod
    def update_velocity(self, prev_state: Grid) -> None:
        """
        Abstract method for updating the velocity. Should be replaced by every implementation of the ABC.
        :param prev_state: previous state of the system to update the velocities on
        """
        pass

    def update_position(self, simulation: Simulation) -> Grid:
        """
        Updates the positions of the vehicles in the lattice. A new grid is created, so the operations can be
        performed while having access to the old state. Vehicle objects do not have a position attribute. Instead,
        they are referenced by Cell objects that make up each slot of the Grid array.

        The method iterates over all the cells, checks whether the cell has a vehicle, and calculates what is the new
        corresponding position. Then, the vehicle object is assigned to the corresponding cell to that position in
        the new grid.

        When vehicles are going to crash, it calculates its final position after the crash, instead of moving them
        "as if" they hadn't crashed.

        :param simulation: Simulation object to perform the update of positions on
        :return: new Grid object with the positions correctly updated (i.e. the vehicles rearranged in the
        corresponding cells).
        """
        prev_state = simulation.state
        new_state: Grid = Grid.new_from(prev_state)
        # Move vehicles
        for x, y in new_state.indices():
            position = Vehicle.Position(x=x, y=y)
            if (vehicle := prev_state.get(position).vehicle) is not None:
                # Based on the updated velocity, the new position is deterministic (i.e. its current position plus
                # the velocity).
                if (vehicle.state.speed > 0
                        and (collision := self.detect_collision(((x, y), vehicle), prev_state)) is not None):
                    # If there is a collision, we halt the vehicle, determine the location and move it there,
                    # and set the remaining collision penalty time to the value provided by
                    # `self.get_collision_penalty`.
                    coll_loc, (_, coll_vehicle) = collision
                    # We set a time penalty to the vehicle that it will have to wait until it can accelerate again.
                    time_penalty = self.get_collision_penalty(vehicle, coll_vehicle)
                    vehicle.state.rct = time_penalty
                    # We add to the collisions counter the severity of this collision, so it can be analysed later.
                    simulation.data.collisions += self.get_collision_weight(vehicle, coll_vehicle)
                    vehicle.state.next_speed = 0
                    new_state.puts_vehicle(coll_loc, vehicle)
                    vehicle.state.collided = True
                    # We add the travelled distance to the vehicle state (this is required for calculating the loss
                    # function).
                    vehicle.state.travelled_distance += np.linalg.norm(coll_loc.to_array() - position.to_array())
                else:
                    # If the position is an allowed position, then the vehicle can be moved there.
                    new_pos = np.array([x, y]) + vehicle.state.velocity
                    if tuple(new_pos) in new_state.indices():
                        # If the new position is in the lattice, we can move the vehicle.
                        new_state.puts_vehicle(Vehicle.Position.from_array(new_pos), vehicle)
                        # We add the travelled distance to the vehicle state (this is required for calculating the loss
                        # function).
                        vehicle.state.travelled_distance += np.linalg.norm(new_pos - position.to_array())
                    else:
                        # Otherwise, the vehicle will no longer be the lattice.
                        # We save the velocity history from the vehicle to the main simulation.
                        simulation.data.speed_histories.append(vehicle.state.speed_history)
                        simulation.data.travelled_distance += vehicle.state.travelled_distance
                        # The vehicle will be garbage-collected as it is no longer being referenced by any object.

        # Since some vehicles might have crashed, their new speed is now in `next_speed`, which has to be merged
        # into the main speed.
        self.merge_speeds(new_state)
        return new_state

    def update_sources(self, prev_state: Grid) -> None:
        """
        Adds vehicles in the edges of the lattice if the cell is a source cell, with a given probability.
        :param prev_state: state to add the vehicles to
        """
        def add_vehicle(cell_, direction):
            """
            We define a closure to add vehicles and avoid repeating code.
            :param cell_: cell to add the vehicle to
            :param direction: direction that the vehicle will be heading to
            :return:
            """
            if cell_.vehicle is None and random.random() < self.traffic_density:
                # We initialise the vehicle
                state = Vehicle.State(direction=direction, speed=self.initial_speed)
                props = Vehicle.Props()
                # We create the vehicle instance
                vehicle = Vehicle(state=state, props=props)
                # We add the vehicle to the cell
                cell_.vehicle = vehicle

        # These are the slices of the array that we will be looking into
        bottom = prev_state[:, 0]
        top = prev_state[:, -1]
        left = prev_state[0, :]
        right = prev_state[-1, :]

        # Add vehicles in different directions for the different cases

        for cell in bottom:
            if (north := Cell.Direction.north) in cell.directions:
                add_vehicle(cell, north)

        for cell in top:
            if (south := Cell.Direction.south) in cell.directions:
                add_vehicle(cell, south)

        for cell in left:
            if (east := Cell.Direction.east) in cell.directions:
                add_vehicle(cell, east)

        for cell in right:
            if (west := Cell.Direction.west) in cell.directions:
                add_vehicle(cell, west)

    @staticmethod
    def get_collision_penalty(vehicle: Vehicle, coll_vehicle: Vehicle) -> float:
        """
        Get collision penalty, i.e. the number of iterations the vehicle will be stopped for.
        This will be saved to the simulation object.
        :param vehicle: vehicle that we are inspecting
        :param coll_vehicle: vehicle it will collide with
        :return: collision penalty
        """
        # The factor of 30 comes from 900 (1h in natural time)
        # over 2 * vmax (which is 15). For more information,
        # see the paper in SS 2.4.1. Eq. 27
        return 30 * np.linalg.norm(vehicle.state.velocity - coll_vehicle.state.velocity)

    @classmethod
    def detect_collision(cls,
                         loc_vehicle: LocatedVehicle,
                         state: Grid,
                         all_colls: bool = False
                         ) -> Optional[Union[Collision, List[Collision]]]:
        """
        Detects whether a collision will happen for a given Grid and provided a specific location of a vehicle.
        :param loc_vehicle: location of the vehicle we are considering collisions for
        :param state: state of the simulation
        :param all_colls: whether to incldue all of the collisions or only the one that will physically happen if
        nothing is changed.
        :return:
        """
        pos, vehicle = loc_vehicle
        (xlo, xhi), (ylo, yhi) = cls.collision_neighbourhood(loc_vehicle)

        # We add one because the boundaries are closed, and the slicing of the array is open
        neighbourhood = state[xlo:xhi + 1, ylo:yhi + 1]
        detected_collisions: List = []
        for (x, y), neighbour_vehicle in Grid.get_vehicles(neighbourhood):
            # We defined the variables found in SS2.4.1
            delta_rx = pos[0] - x
            delta_ry = pos[1] - y
            delta_vx = neighbour_vehicle.state.velocity[0] - vehicle.state.velocity[0]
            delta_vy = neighbour_vehicle.state.velocity[1] - vehicle.state.velocity[1]

            # Case where both differences in velocity components are non-null
            if delta_vy == 0 and delta_vx == 0:
                continue
            elif delta_vy != 0 and delta_vx != 0:
                f_range = (delta_vy * delta_rx - abs(delta_vy), delta_vy * delta_rx + abs(delta_vy))
                g_range = (delta_vx * delta_ry - abs(delta_vx), delta_vx * delta_ry + abs(delta_vx))
                # If the intersection of both sets is non-null there is a crash
                if ((f_range[0] <= g_range[0] < f_range[1]) and g_range[1] > g_range[0]
                        or (g_range[0] <= f_range[0] < g_range[1]) and f_range[1] > g_range[0]):
                    # Here, we calculate the minimum of k/(deltavy*deltavx)
                    # To do that, we consider all the boundaries, and obtain the intersection of the sets.
                    # Then, we calculate the extrema of the values of t (k is monotonous)
                    # We pick the minimum value from those calculated and use that as the timespan to get the
                    # position of collision.
                    bounds = sorted([f_range[0], f_range[1], g_range[0], g_range[1]])
                    t = np.array(bounds[1:3]) / (delta_vy * delta_vx)
                    r_collision = pos + min(list(t)) * vehicle.state.velocity
                else:
                    r_collision = None
            # Cases where either of the differences in velocity components is null. See SS2.4.1
            elif delta_vy == 0 and np.sign(delta_vx) == np.sign(delta_rx) and -1 < delta_ry < 1:
                r_collision = pos + np.sign(delta_rx) * (abs(delta_rx) - 1) / delta_vx * vehicle.state.velocity
            elif delta_vx == 0 and np.sign(delta_vy) == np.sign(delta_rx) and -1 < delta_rx < 1:
                r_collision = pos + np.sign(delta_ry) * (abs(delta_ry) - 1) / delta_vy * vehicle.state.velocity
            else:
                r_collision = None
            if r_collision is not None:
                # We round the collision spot to the closest cell
                r_collision_cell = np.array((int(r_collision[0]), int(r_collision[1])), dtype=int)
                # We append the collision spot to the list, to later calculate which collision occurs first.
                detected_collisions.append((r_collision_cell, ((x, y), neighbour_vehicle)))
        if len(detected_collisions) > 0:
            if all_colls:
                # If all collisions are requested, we retrieve them all.
                return [(Vehicle.Position.from_array(collision[0]), collision[1]) for collision in detected_collisions]
            # Otherwise, we calculate which of the points is closer, and retrieve it
            collision = min(detected_collisions, key=lambda el: np.linalg.norm(el[0] - pos))
            return Vehicle.Position.from_array(collision[0]), collision[1]
        else:
            return None

    @staticmethod
    def collision_neighbourhood(loc_vehicle: LocatedVehicle) -> Tuple[Tuple, Tuple]:
        """
        Gets the neighbourhood of a given cell to determine whether it will collide with other vehicles.
        :param loc_vehicle: the location of the vehicle to obtain the neighbourhood slice
        :return: boundaries of the grid to perform the slice on, and obtain the collision neighbourhood. it returns a
        (x_lower, x_upper), (y_lower, y_upper) tuple.
        """
        pos, vehicle = loc_vehicle
        # Unitary vector for velocity
        uvel = vehicle.state.uvel
        # Perpendicular unitary vector of velocity
        puvel = np.array((-uvel[1], uvel[0]))
        # We convert the position to a numpy array
        pos = np.array(pos)

        # These are the anchor points, i.e. the position vectors
        # of the edges of the neighbourhood
        anchors = (
            pos - Vehicle.max_speed * puvel,  # Close, Left
            pos + Vehicle.max_speed * puvel,  # Close, Right
            pos - Vehicle.max_speed * puvel + vehicle.state.velocity,  # Far, Left
            pos + Vehicle.max_speed * puvel + vehicle.state.velocity  # Far, Right
        )
        # We extract the coordinates of the anchor points to construct the slicing range
        xbounds = sorted(list(set([i[0] for i in anchors])))
        if len(xbounds) == 1:
            # This means that both the start and end values are equal
            xbounds = [xbounds[0], xbounds[0]]
        ybounds = sorted(list(set([i[1] for i in anchors])))
        if len(ybounds) == 1:
            ybounds = [ybounds[0], ybounds[0]]
        return tuple(max(0, int(el)) for el in xbounds), tuple(max(0, int(el)) for el in ybounds)

    @staticmethod
    def merge_speeds(state: Grid):
        """
        Updates the speeds of the vehicles, copying the value of `next_speed` onto `speed` and then clearing
        `next_speed`.
        :param state: the state to consider
        """
        for x, y in state.indices():
            if (vehicle := state[x, y].vehicle) is not None and vehicle.state.next_speed is not None:
                vehicle.state.speed = vehicle.state.next_speed
                vehicle.state.next_speed = None

    @staticmethod
    def get_collision_weight(vehicle_1: Vehicle, vehicle_2: Vehicle) -> float:
        """
        Gets a value that is part of the expression of the loss function. This value represents the severity of a
        collision, and is to be stored to then be analysed later.
        :param vehicle_1: vehicle 1 that is to collide
        :param vehicle_2: vehicle 2 that is to collide
        :return:
        """
        # Compared to the formula in the paper, we device by two since we are counting each collision twice
        # (for each vehicle we perform the collision check and add the value)
        return np.linalg.norm(vehicle_1.state.velocity - vehicle_2.state.velocity) / (2 * Vehicle.max_speed)

    @staticmethod
    def minimum_vel_to_prevent_collision(
            vehicle: Vehicle, vehicle_location: np.ndarray, collision_location: np.ndarray) -> int:
        """
        Gets the minimum velocity that a vehicle can adopt when it is trying to prevent a collision, following the
        rules specified in the paper. This will be either the speed required to be just one cell before the collision
        scenario, or the minimum speed it can decelerate to given its current speed, whichever is lower.
        :param vehicle: vehicle to consider a deceleration for
        :param vehicle_location: the location of the vehicle in the grid
        :param collision_location: the location where it would collide
        :return:
        """
        delta_r = collision_location - vehicle_location
        if np.linalg.norm(delta_r) == 0:
            desired_vel = np.array([0, 0])
        else:
            desired_vel = delta_r - vehicle.state.uvel
        # Deceleration required to prevent the crash
        required_deceleration = np.linalg.norm(desired_vel - vehicle.state.velocity)
        if required_deceleration > Vehicle.max_deceleration:
            return int(vehicle.state.speed - Vehicle.max_deceleration)
        else:
            return int(np.linalg.norm(desired_vel))

    @staticmethod
    def can_prevent_collision(
            vehicle: Vehicle, vehicle_location: np.ndarray, collision_location: np.ndarray) -> bool:
        """
        Determines whether the vehicle is able to decelerate enough within its physical limits to prevent a collision.
        :param vehicle: vehicle to consider a deceleration for
        :param vehicle_location: the location of the vehicle in the grid
        :param collision_location: the location where it would collide
        :return:
        """
        delta_r = collision_location - vehicle_location
        if np.linalg.norm(delta_r) == 0:
            desired_vel = np.array([0, 0])
        else:
            desired_vel = delta_r - vehicle.state.uvel
        # Deceleration required to prevent the crash
        required_deceleration = np.linalg.norm(desired_vel - vehicle.state.velocity)
        return required_deceleration <= Vehicle.max_deceleration


class Policy0(Policy):
    """
    Policy with no extra rules (i.e. vehicles simply crash).
    """
    def update_velocity(self, prev_state: Grid):
        for x, y in prev_state.indices():
            if (vehicle := prev_state[x, y].vehicle) is not None:
                # We save the current velocity to the velocity history.
                vehicle.state.speed_history.append(vehicle.state.speed)
                # Pi_0 copies v(t-1) to v(t)
                vehicle.state.next_speed = vehicle.state.speed

        # Once all the speeds have been calculated, we can merge the changes back into the current speed.
        self.merge_speeds(prev_state)


class Policy1(Policy):
    """
    A basic policy that tries to stop if another vehicle is going to crash with it.
    """
    def update_velocity(self, prev_state: Grid) -> None:
        for x, y in prev_state.indices():
            if (vehicle := prev_state[x, y].vehicle) is not None:
                # We save the current velocity to the velocity history.
                vehicle.state.speed_history.append(vehicle.state.speed)

                # Current Vehicle Position
                current_vehicle_pos = Vehicle.Position(x=x, y=y)

                if vehicle.state.collided:
                    if vehicle.state.rct > 0:
                        vehicle.state.rct -=1
                        continue
                    else:
                        vehicle.state.collided = False

                # If the vehicle will collide:
                if (vehicle.state.speed > 0
                        and (collision := self.detect_collision(((x, y), vehicle), prev_state)) is not None):
                    # Collision detected
                    collision: Collision
                    # Location of the collision, and the vehicle it will collide with
                    coll_location, _ = collision
                    vehicle.state.next_speed = self.minimum_vel_to_prevent_collision(
                        vehicle=vehicle,
                        collision_location=coll_location.to_array(),
                        vehicle_location=current_vehicle_pos.to_array()
                    )
                else:
                    # No collision detected
                    # We will now see if we can increase its speed by 1 (if it does not collide if so).
                    # In order to check whether the vehicle would collide, we tweak the vehicle speed to that speed
                    # we want to check for. After that, the vehicle is rolled back (so it doesn't affect the check
                    # for other vehicles in the lattice) and we set the next_speed accordingly.
                    #
                    # We could also create a copy of the object, but that would require eliminating its own position
                    # from the detect_collision check, or else the function would fail.
                    vehicle.state.speed += 1
                    collision = self.detect_collision(((x, y), vehicle), prev_state)
                    vehicle.state.speed -= 1
                    if collision is None:
                        # Can accelerate
                        vehicle.state.next_speed = vehicle.state.speed + 1
                    else:
                        # Cannot accelerate
                        vehicle.state.next_speed = vehicle.state.speed
        self.merge_speeds(prev_state)


class Policy2(Policy):
    """
    A policy algorithm that uses a collision graph to detect priority of deceleration, and stops vehicles with a
    higher number of potential collisions, to facilitate the flow of vehicles.
    """
    def update_velocity(self, prev_state: Grid) -> None:
        # Calculate collision graph
        collision_graph: List[Tuple[LocatedVehicle, LocatedVehicle]] = []
        for x, y in prev_state.indices():
            if (vehicle := prev_state[x, y].vehicle) is not None:

                # Vehicles that have collided and are halted to not need to be accounted for.
                if vehicle.state.collided: continue

                # We save the current velocity to the velocity history.
                vehicle.state.speed_history.append(vehicle.state.speed)
                if (vehicle := prev_state[x, y].vehicle) is not None:
                    collisions = self.detect_collision(((x, y), vehicle), prev_state, all_colls=True)
                    if collisions is not None:
                        for collision in collisions:
                            collision: Collision
                            # We extract the collision location and the located vehicle
                            coll_location, located_neighbour = collision
                            if self.can_prevent_collision(
                                vehicle=vehicle,
                                collision_location=coll_location.to_array(),
                                vehicle_location=np.array((x, y))
                            ):
                                collision_graph.append((
                                    ((x, y), vehicle),
                                    located_neighbour,
                                ))

        # Obtain nodes and their number of collisions
        # The nodes are represented by their coordinates
        nodes = set([edge[0][0] for edge in collision_graph])

        def count_ocurrences(node):
            i = 0
            for edge in collision_graph:
                if edge[0][0] == node:
                    i += 1
            return i

        selected_nodes = []
        while len(collision_graph) > 0:
            node_counts = sorted([(node, count_ocurrences(node)) for node in nodes], key=lambda el: -el[1])
            max_count = node_counts[0][1]
            selected_node = random.choice(list(filter(lambda el: el[1] == max_count, node_counts)))[0]
            selected_nodes.append(selected_node)
            # Remove the element from the graph, as well as all the other reciprocal edges
            # First we exclude the edges of the form (e, w) where e is the selected edge
            # Then, we will also exclude those edges that have their reciprocals.
            # This is because if the edge is bi-directional, the other edge is no longer a risk when the one edge is
            # removed. However, if one edge is one-directional, just because the vehicle is stopped, it does not mean
            # that the risk will disappear (imagine two vehicles in a crossroad approaching each other vs. one
            # vehicle going too fast behind another).
            possible_reciprocals = [edge[::-1] for edge in filter(lambda edge: edge[0][0] == selected_node, collision_graph)]
            collision_graph = list(filter(
                lambda edge: edge[0][0] != selected_node and edge not in possible_reciprocals,
                collision_graph
            ))

        for x, y in prev_state.indices():
            if (vehicle := prev_state[x, y].vehicle) is not None:
                if vehicle.state.collided:
                    if vehicle.state.rct > 0:
                        vehicle.state.rct -= 1
                        continue
                    else:
                        vehicle.state.collided = False
                if (x, y) in selected_nodes:
                    coll_loc = self.detect_collision(((x,y), vehicle), state=prev_state)[0]
                    vehicle.state.next_speed = self.minimum_vel_to_prevent_collision(vehicle, np.array((x,y)),
                                                                                     coll_loc.to_array())
                else:
                    vehicle.state.speed += 1
                    collision = self.detect_collision(((x, y), vehicle), prev_state)
                    vehicle.state.speed -= 1
                    if collision is None:
                        # Can accelerate
                        vehicle.state.next_speed = vehicle.state.speed + 1
                    else:
                        # Cannot accelerate
                        vehicle.state.next_speed = vehicle.state.speed

        self.merge_speeds(prev_state)
