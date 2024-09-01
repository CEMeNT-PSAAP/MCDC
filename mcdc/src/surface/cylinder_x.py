"""
Cylinder-X: Infinite cylinder parallel to the x-axis
    f(y, z) = y^2 + z^2 + H * y + I * z + J
"""

from numba import njit

import mcdc.src.surface.common as common


@njit
def evaluate(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    """
    # Particle parameters
    y = particle["y"]
    z = particle["z"]

    # Surface parameters
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    return y**2 + z**2 + H * y + I * z + J
