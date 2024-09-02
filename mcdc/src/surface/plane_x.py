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

    return particle["x"] + surface["J"]


@njit
def evaluate_moving(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate [Moving version]

    f(x(t)) = x(t) + J, and x(t) = x0 - Vx * t,
    where Vx is the x-component of the move translation velocity.
    """
    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Translation velocity
    Vx = surface["move_velocities"][idx][0]

    # Translated position
    t_local = particle["t"] - surface["move_time_grid"][idx]
    x_translated = particle["x"] - Vx * t_local

    return x_translated + surface["J"]


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
    Particle speed is needed if the surface is moving to get the relative direction.
    """
    if surface["moving"]:
        return get_normal_component_moving(particle, speed, surface)

    return particle["ux"]


@njit
def get_normal_component_moving(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle [Moving version]

    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative direction.
    """
    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Translation velocity
    Vx = surface["move_velocities"][idx][0]

    # Return relative direction
    return particle["ux"] - Vx / speed


@njit
def get_distance(particle, speed, surface):
    """
    Get particle distance to surface
    """
    if surface["moving"]:
        return get_distance_moving(particle, speed, surface)

    # Parallel?
    normal_component = get_normal_component(particle, speed, surface)
    if abs(normal_component) == 0.0:
        return INF

    # Coincident?
    f = evaluate(particle, surface)
    if abs(f) < COINCIDENCE_TOLERANCE:
        return INF

    # Calculate distance
    distance = -f / normal_component

    # Moving away?
    if distance < 0.0:
        return INF

    return distance


@njit
def get_distance_moving(particle, speed, surface):
    """
    Get particle distance to surface [Moving version]
    """
    # Store particle coordinate (will be temporarily changed)
    x_original = particle["x"]
    ux_original = particle["ux"]
    t_original = particle["t"]
    surface["moving"] = False

    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Distance accumulator
    total_distance = 0.0

    # Evaluate the current and the subsequent intervals until intersecting
    while idx < surface["N_move"]:
        # Apply translation velocity
        Vx = surface["move_velocities"][idx][0]
        particle["ux"] -= Vx / speed

        # Get distance
        distance = get_distance(particle, speed, surface)

        # Beyond the interval?
        distance_time = distance / speed
        dt = surface["move_time_grid"][idx + 1] - particle["t"]
        if distance_time > dt:
            distance = INF

        # Intersecting?
        if distance < INF:
            # Restore particle coordinate
            particle["x"] = x_original
            particle["ux"] = ux_original
            particle["t"] = t_original
            surface["moving"] = False

            # Return the total distance
            return total_distance + distance

        # Accumulate distance
        additional_distance = dt * speed
        total_distance += additional_distance

        # Modify the particle
        particle["x"] += additional_distance * ux_original
        particle["ux"] = ux_original
        particle["t"] = surface["move_time_grid"][idx + 1]

        # Check next interval
        idx += 1

    # Restore particle coordinate
    particle["x"] = x_original
    particle["ux"] = ux_original
    particle["t"] = t_original
    surface["moving"] = False

    # No intersection
    return INF
