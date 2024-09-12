"""
Plane-Z: Plane perpendicular to the z-axis

f(z) = z + J
"""

from numba import njit

import mcdc.src.surface.common as common

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)


@njit
def evaluate(particle, surface):
    return particle["z"] + surface["J"]


@njit
def reflect(particle, surface):
    particle["uz"] = -particle["uz"]


@njit
def get_normal_component(particle, surface):
    return particle["uz"]


@njit
def get_distance(particle, surface):
    # Parallel?
    normal_component = get_normal_component(particle, surface)
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
