"""
Plane-X: Plane perpendicular to the x-axis

f(x) = x + J
"""

from numba import njit

import mcdc.src.surface.common as common

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)


@njit
def evaluate(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    """
    if surface["moving"]:
        return evaluate_moving(particle, surface)

    # Particle and surface parameters
    x = particle["x"]
    J = surface["J"]

    return x + J


@njit
def evaluate_moving(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate [Moving version]

    f(x(t)) = x(t) + J, and x(t) = x0 - Vx * t,
    where Vx is the x-component of the move translation velocity.
    """
    # Particle and surface parameters
    x = particle["x"]
    t = particle["t"]
    J = surface["J"]

    # Move interval index
    idx = common._get_move_idx(t, surface)

    # Translation velocity
    x1 = surface["move_translations"][idx][0]
    x2 = surface["move_translations"][idx + 1][0]
    t1 = surface["move_time_grid"][idx]
    t2 = surface["move_time_grid"][idx + 1]
    Vx = (x2 - x1) / (t2 - t1)

    # Translate the position
    t_local = t - t1
    x = x - Vx * t_local

    return x + J


@njit
def reflect(particle, surface):
    """
    Reflect the particle off the surface
    """
    particle["ux"] = -particle["ux"]


@njit
def get_normal_component(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle

    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative velocity.
    """
    if surface["moving"]:
        return get_normal_component_moving(particle, speed, surface)

    return particle["ux"]


@njit
def get_normal_component_moving(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle [Moving version]

    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative velocity.
    """
    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Translation velocity
    x1 = surface["move_translations"][idx][0]
    x2 = surface["move_translations"][idx + 1][0]
    t1 = surface["move_time_grid"][idx]
    t2 = surface["move_time_grid"][idx + 1]
    Vx = (x2 - x1) / (t2 - t1)

    # Return relative velocity
    return particle["ux"] - Vx / speed


@njit
def get_distance(particle, speed, surface):
    """
    Get particle distance to surface
    """
    if surface["moving"]:
        return get_distance_moving(particle, speed, surface)

    # Particle and surface parameters
    x = particle["x"]
    ux = particle["ux"]
    J = surface["J"]

    # Parallel?
    if ux == 0.0:
        return INF

    # Coincident?
    if abs(x + J) < COINCIDENCE_TOLERANCE:
        return INF

    # Calculate distance
    distance = -(x + J) / ux

    # Moving away?
    if distance < 0.0:
        return INF

    return distance


@njit
def get_distance_moving(particle, speed, surface):
    """
    Get particle distance to surface [Moving version]
    """
    # Particle and surface parameters
    x = particle["x"]
    ux = particle["ux"]
    t = particle["t"]
    J = surface["J"]

    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Coincident?
    coincident = abs(particle["x"] - J) < COINCIDENCE_TOLERANCE

    # ==============================================================================
    # Evaluate starting interval if not coincident

    if not coincident:
        # Translation velocity
        x1 = surface["move_translations"][idx][0]
        x2 = surface["move_translations"][idx + 1][0]
        t1 = surface["move_time_grid"][idx]
        t2 = surface["move_time_grid"][idx + 1]
        Vx = (x2 - x1) / (t2 - t1)

        # The relative velocity
        relative_velocity = ux - Vx / speed

        # Check if particle and surface NOT move in the same velocity and never meet
        if relative_velocity != 0.0:
            distance = -(x + J) / relative_velocity
            # Check if not moving away
            if distance > 0.0:
                # Check if it is still within the interval
                distance_time = distance / speed
                if distance_time <= time_grid[idx + 1] - t:
                    return distance

    # Not intersecting in the starting interval. Let's check the next ones
    idx += 1

    # But first, we need to keep track of the total distance traveled and
    # the particle position (which we temporarily change)
    total_distance = (time_grid[idx] - t) * speed
    particle["x"] += total_distance * ux

    # ==============================================================================
    # Evaluate subsequent interval

    while idx < N_move:
        # Translation velocity
        x1 = surface["move_translations"][idx][0]
        x2 = surface["move_translations"][idx + 1][0]
        t1 = surface["move_time_grid"][idx]
        t2 = surface["move_time_grid"][idx + 1]
        dt = t2 - t1
        Vx = (x2 - x1) / dt

        # The relative velocity
        relative_velocity = ux - Vx / speed

        # Check if particle and surface NOT move in the same velocity and never meet
        if relative_velocity != 0.0:
            distance = -(x + J) / relative_velocity
            # Check if not moving away
            if distance > 0.0:
                # Check if it is still within the interval
                distance_time = distance / speed
                if distance_time <= dt:
                    # Restore particle coordinate
                    particle["x"] = x
                    return total_distance + distance

        # Not intersecting within the current interval
        additional_distance = dt * speed
        total_distance += additional_distance
        particle["x"] += additional_distance * ux

        idx += 1

    # No intersection
    distance = INF

    # Restore particle coordinate
    particle["x"] = x

    return INF
