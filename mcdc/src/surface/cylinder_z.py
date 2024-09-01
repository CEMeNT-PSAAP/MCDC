"""
Cylinder-Z: Infinite cylinder parallel to the z-axis
    f(x, y) = x^2 + y^2 + G * x + H * y + J
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

    # Surface parameters
    G = surface["G"]
    H = surface["H"]
    J = surface["J"]

    return x**2 + y**2 + G * x + H * y + J
