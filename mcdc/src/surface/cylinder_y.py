"""
Cylinder-Y: Infinite cylinder parallel to the y-axis

f(x, z) = xx + zz + Gx + Iz + J
"""

import math

from numba import njit

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)


@njit
def evaluate(particle, surface):
    # Particle parameters
    x = particle["x"]
    z = particle["z"]

    # Surface parameters
    G = surface["G"]
    I = surface["I"]
    J = surface["J"]

    return x**2 + z**2 + G * x + I * z + J


@njit
def reflect(particle, surface):
    # Particle parameters
    ux = particle["ux"]
    uz = particle["uz"]

    # Surface normal
    dx = 2 * particle["x"] + surface["G"]
    dz = 2 * particle["z"] + surface["I"]
    norm = (dx**2 + dz**2) ** 0.5
    nx = dx / norm
    nz = dz / norm

    # Reflect
    c = 2.0 * (nx * ux + nz * uz)
    particle["ux"] -= c * nx
    particle["uz"] -= c * nz


@njit
def get_normal_component(particle, surface):
    # Surface normal
    dx = 2 * particle["x"] + surface["G"]
    dz = 2 * particle["z"] + surface["I"]
    norm = (dx**2 + dz**2) ** 0.5
    nx = dx / norm
    nz = dz / norm

    # Particle parameters
    ux = particle["ux"]
    uz = particle["uz"]

    return nx * ux + nz * uz


@njit
def get_distance(particle, surface):
    # Particle coordinate
    x = particle["x"]
    z = particle["z"]
    ux = particle["ux"]
    uz = particle["uz"]

    # Surface coefficients
    G = surface["G"]
    I = surface["I"]

    # Coincident?
    f = evaluate(particle, surface)
    coincident = abs(f) < COINCIDENCE_TOLERANCE
    if coincident:
        # Moving away or tangent?
        if get_normal_component(particle, surface) >= 0.0 - COINCIDENCE_TOLERANCE:
            return INF

    # Quadratic equation constants
    a = ux * ux + uz * uz
    b = 2 * (x * ux + z * uz) + G * ux + I * uz
    c = f

    determinant = b * b - 4.0 * a * c

    # Roots are complex : no intersection
    # Roots are identical: tangent
    # ==> return huge number
    if determinant <= 0.0:
        return INF
    else:
        # Get the roots
        denom = 2.0 * a
        sqrt = math.sqrt(determinant)
        root_1 = (-b + sqrt) / denom
        root_2 = (-b - sqrt) / denom

        # Coincident?
        if coincident:
            return max(root_1, root_2)

        # Negative roots, moving away from the surface
        if root_1 < 0.0:
            root_1 = INF
        if root_2 < 0.0:
            root_2 = INF

        # Return the smaller root
        return min(root_1, root_2)
