"""
Plane-Z: Plane perpendicular to the z-axis
    f(z) = z + J
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
    z = particle["z"]
    J = surface["J"]

    return z + J


@njit
def evaluate_moving(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    [Moving version]
    f(z(t)) = z(t) + J, and z(t) = z0 - Vz * t,
    where Vz is the z-component of the move translation velocity.
    """
    # Particle and surface parameters
    z = particle["z"]
    t = particle["t"]
    J = surface["J"]

    # Move interval index
    idx = common._get_move_idx(t, surface)

    # Translation edge points and the associated time
    z1 = surface["move_translations"][idx][2]
    z2 = surface["move_translations"][idx + 1][2]
    t1 = surface["move_time_grid"][idx]
    t2 = surface["move_time_grid"][idx + 1]

    # Translation velocity
    Vz = (z2 - z1) / (t2 - t1)

    # Translate the position
    t_local = t - t1
    z = z - Vz * t_local

    return z + J
