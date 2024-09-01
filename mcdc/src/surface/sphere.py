"""
Sphere
    f(x, y, z) = x^2 + y^2 + z^2 + G * x + H * y + I * z + J
"""

from numba import njit

import mcdc.src.surface.common as common


@njit
def evaluate(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    """
    # Particle parameters
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]

    # Surface parameters
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    return x**2 + y**2 + z**2 + G * x + H * y + I * z + J
