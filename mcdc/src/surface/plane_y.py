"""
Plane-Y: Plane perpendicular to the y-axis

f(y) = y + J
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
        return _evaluate_moving(particle, surface)

    return particle["y"] + surface["J"]


@njit
def _evaluate_moving(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate [Moving version]

    f(y(t)) = y(t) + J, and y(t) = y0 - Vy * t,
    where Vy is the y-component of the move translation velocity.
    """
    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Translation velocity
    Vy = surface["move_velocities"][idx, 1]

    # Translated position
    t_local = particle["t"] - surface["move_time_grid"][idx]
    y_translated = particle["y"] - Vy * t_local

    return y_translated + surface["J"]


@njit
def reflect(particle, surface):
    """
    Reflect the particle off the surface
    """
    particle["uy"] = -particle["uy"]


@njit
def get_normal_component(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle

    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative direction.
    """
    if surface["moving"]:
        return _get_normal_component_moving(particle, speed, surface)

    return particle["uy"]


@njit
def _get_normal_component_moving(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle [Moving version]

    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative direction.
    """
    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Translation velocity
    Vy = surface["move_velocities"][idx, 1]

    # Return relative direction
    return particle["uy"] - Vy / speed


@njit
def get_distance(particle, speed, surface):
    """
    Get particle distance to surface
    """
    if surface["moving"]:
        return _get_distance_moving(particle, speed, surface)

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
def _get_distance_moving(particle, speed, surface):
    """
    Get particle distance to surface [Moving version]
    """
    # Store particle coordinate (will be temporarily changed)
    y_original = particle["y"]
    uy_original = particle["uy"]
    t_original = particle["t"]
    surface["moving"] = False

    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Distance accumulator
    total_distance = 0.0

    # Evaluate the current and the subsequent intervals until intersecting
    while idx < surface["N_move"]:
        # Apply translation velocity
        Vy = surface["move_velocities"][idx, 1]
        particle["uy"] -= Vy / speed

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
            particle["y"] = y_original
            particle["uy"] = uy_original
            particle["t"] = t_original
            surface["moving"] = False

            # Return the total distance
            return total_distance + distance

        # Accumulate distance
        additional_distance = dt * speed
        total_distance += additional_distance

        # Modify the particle
        particle["y"] += additional_distance * uy_original
        particle["uy"] = uy_original
        particle["t"] = surface["move_time_grid"][idx + 1]

        # Check next interval
        idx += 1

    # Restore particle coordinate
    particle["y"] = y_original
    particle["uy"] = uy_original
    particle["t"] = t_original
    surface["moving"] = False

    # No intersection
    return INF
