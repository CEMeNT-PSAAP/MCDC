"""
Plane-X: Plane perpendicular to the x-axis
    f(x) = x + J
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
    x = particle["x"]
    J = surface["J"]

    return x + J


@njit
def evaluate_moving(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    [Moving version]
    f(x(t)) = x(t) + J, and x(t) = x0 - Vx * t,
    where Vx is the x-component of the move translation velocity.
    """
    # Particle and surface parameters
    x = particle["x"]
    t = particle["t"]
    J = surface["J"]

    # Move interval index
    idx = common._get_move_idx(t, surface)

    # Translation edge points and the associated time
    x1 = surface["move_translations"][idx][0]
    x2 = surface["move_translations"][idx + 1][0]
    t1 = surface["move_time_grid"][idx]
    t2 = surface["move_time_grid"][idx + 1]

    # Translation velocity
    Vx = (x2 - x1) / (t2 - t1)

    # Translate the position
    t_local = t - t1
    x = x - Vx * t_local

    return x + J
