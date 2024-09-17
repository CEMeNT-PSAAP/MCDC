"""
Plane-X: Plane perpendicular to the x-axis

f(x) = x + J
"""

from numba import njit

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)


@njit
def evaluate(particle, surface):
    return particle["x"] + surface["J"]


@njit
def reflect(particle, surface):
    particle["ux"] = -particle["ux"]


@njit
def get_normal_component(particle, surface):
    return particle["ux"]


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
