"""
Cylinder-Y: Infinite cylinder parallel to the y-axis
    f(x, z) = x^2 + z^2 + G * x + I * z + J
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
    z = particle["z"]

    # Surface parameters
    G = surface["G"]
    I = surface["I"]
    J = surface["J"]

    return x**2 + z**2 + G * x + I * z + J
