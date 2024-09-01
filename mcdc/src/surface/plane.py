"""
Plane:
    f(x, y, z) = G * x + H * y + I * z + J
"""

from numba import njit

import mcdc.src.surface.common as common


@njit
def evaluate(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    """
    if surface["moving"]:
        return evaluate_moving(particle, surface)

    # Particle parameters
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]

    # Surface parameters
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    return G * x + H * y + I * z + J


@njit
def evaluate_moving(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    [Moving version]
    f(x(t), y(t), z(t)) = G * x(t) + H * y(t) + I * z(t) + J,
    and
        x(t) = x0 - Vx * t,
        y(t) = y0 - Vy * t,
        z(t) = z0 - Vz * t,
    where V is the move translation velocity.
    """
    # Particle parameters
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]

    # Surface parameters
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    # Move interval index
    idx = common._get_move_idx(t, surface)

    # Translation edge points and the associated time
    x1 = surface["move_translations"][idx][0]
    x2 = surface["move_translations"][idx + 1][0]
    y1 = surface["move_translations"][idx][1]
    y2 = surface["move_translations"][idx + 1][1]
    z1 = surface["move_translations"][idx][2]
    z2 = surface["move_translations"][idx + 1][2]
    t1 = surface["move_time_grid"][idx]
    t2 = surface["move_time_grid"][idx + 1]

    # Translation velocity
    Vx = (x2 - x1) / (t2 - t1)
    Vy = (y2 - y1) / (t2 - t1)
    Vz = (z2 - z1) / (t2 - t1)

    # Translate the position
    t_local = t - t1
    x = x - Vx * t_local
    y = y - Vy * t_local
    z = z - Vz * t_local

    return G * x + H * y + I * z + J
