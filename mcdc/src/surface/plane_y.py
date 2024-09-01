"""
Plane-Y: Plane perpendicular to the y-axis
    f(y) = y + J
"""

from numba import njit

import mcdc.src.surface.common as common


@njit
def evaluate(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    """
    if surface["moving"]:
        evaluate_moving(particle, surface)

    # Particle and surface parameters
    y = particle["y"]
    J = surface["J"]

    return y + J


@njit
def evaluate_moving(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    [Moving version]
    f(y(t)) = y(t) + J, and y(t) = y0 - Vy * t,
    where Vy is the y-component of the move translation velocity.
    """
    # Particle and surface parameters
    y = particle["y"]
    t = particle["t"]
    J = surface["J"]

    # Move interval index
    idx = common._get_move_idx(t, surface)

    # Translation edge points and the associated time
    y1 = surface["move_translations"][idx][1]
    y2 = surface["move_translations"][idx + 1][1]
    t1 = surface["move_time_grid"][idx]
    t2 = surface["move_time_grid"][idx + 1]

    # Translation velocity
    Vy = (y2 - y1) / (t2 - t1)

    # Translate the position
    t_local = t - t1
    y = y - Vy * t_local

    return y + J
