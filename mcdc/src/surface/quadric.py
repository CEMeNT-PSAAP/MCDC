"""
Quadric
    f(x, y, z) = A * x^2 + B * y^2 + C * z^2 + D * x * y + E * x * z + F * y * z + G * x + H * y + I * z + J
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
    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    return (
        A * x * x
        + B * y * y
        + C * z * z
        + D * x * y
        + E * x * z
        + F * y * z
        + G * x
        + H * y
        + I * z
        + J
    )
