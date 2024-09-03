"""
Surface operations based on the quadric equation:
   f(x,y,z) = Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J
"""

import math

from numba import njit

import mcdc.src.surface.plane_x as plane_x
import mcdc.src.surface.plane_y as plane_y
import mcdc.src.surface.plane_z as plane_z
import mcdc.src.surface.plane as plane
import mcdc.src.surface.cylinder_x as cylinder_x
import mcdc.src.surface.cylinder_y as cylinder_y
import mcdc.src.surface.cylinder_z as cylinder_z
import mcdc.src.surface.sphere as sphere
import mcdc.src.surface.quadric as quadric

from mcdc.algorithm import binary_search_with_length
from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
    SURFACE_LINEAR,
    SURFACE_QUADRIC,
    SURFACE_PLANE_X,
    SURFACE_PLANE_Y,
    SURFACE_PLANE_Z,
    SURFACE_PLANE,
    SURFACE_CYLINDER_X,
    SURFACE_CYLINDER_Y,
    SURFACE_CYLINDER_Z,
    SURFACE_SPHERE,
    SURFACE_QUADRIC,
)


@njit
def check_sense(particle, speed, surface):
    """
    Check on which side of the surface the particle is
        - Return True if on positive side
        - Return False otherwise
    Particle direction and speed are used to tiebreak coincidence.
    """
    result = evaluate(particle, surface)

    # Check if coincident on the surface
    if abs(result) < COINCIDENCE_TOLERANCE:
        # Determine sense based on the direction
        return get_normal_component(particle, speed, surface) > 0.0

    return result > 0.0


@njit
def evaluate(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    """
    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            return plane_x.evaluate(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            return plane_y.evaluate(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            return plane_z.evaluate(particle, surface)
        elif surface["type"] & SURFACE_PLANE:
            return plane.evaluate(particle, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            return cylinder_x.evaluate(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            return cylinder_y.evaluate(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            return cylinder_z.evaluate(particle, surface)
        elif surface["type"] & SURFACE_SPHERE:
            return sphere.evaluate(particle, surface)
        else:
            return quadric.evaluate(particle, surface)


@njit
def get_normal_component(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle
    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative direction.
    """
    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            return plane_x.get_normal_component(particle, speed, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            return plane_y.get_normal_component(particle, speed, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            return plane_z.get_normal_component(particle, speed, surface)
        elif surface["type"] & SURFACE_PLANE:
            return plane.get_normal_component(particle, speed, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            return cylinder_x.get_normal_component(particle, speed, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            return cylinder_y.get_normal_component(particle, speed, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            return cylinder_z.get_normal_component(particle, speed, surface)
        elif surface["type"] & SURFACE_SPHERE:
            return sphere.get_normal_component(particle, speed, surface)
        else:
            return quadric.get_normal_component(particle, speed, surface)


@njit
def reflect(particle, surface):
    """
    Reflect the particle off the surface
    """
    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            return plane_x.reflect(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            return plane_y.reflect(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            return plane_z.reflect(particle, surface)
        elif surface["type"] & SURFACE_PLANE:
            return plane.reflect(particle, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            return cylinder_x.reflect(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            return cylinder_y.reflect(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            return cylinder_z.reflect(particle, surface)
        elif surface["type"] & SURFACE_SPHERE:
            return sphere.reflect(particle, surface)
        else:
            return quadric.reflect(particle, surface)


@njit
def get_distance(particle, speed, surface):
    """
    Get particle distance to surface
    """
    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            return plane_x.get_distance(particle, speed, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            return plane_y.get_distance(particle, speed, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            return plane_z.get_distance(particle, speed, surface)
        elif surface["type"] & SURFACE_PLANE:
            return plane.get_distance(particle, speed, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            return cylinder_x.get_distance(particle, speed, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            return cylinder_y.get_distance(particle, speed, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            return cylinder_z.get_distance(particle, speed, surface)
        elif surface["type"] & SURFACE_SPHERE:
            return sphere.get_distance(particle, speed, surface)
        else:
            return quadric.get_distance(particle, speed, surface)


# ======================================================================================
# Private
# ======================================================================================


@njit
def _get_move_idx(t, surface):
    N_move = surface["N_move"]
    time_grid = surface["move_time_grid"]
    idx = binary_search_with_length(t, time_grid, N_move)

    # Coinciding cases
    if abs(time_grid[idx + 1] - t) < COINCIDENCE_TOLERANCE:
        idx += 1

    return idx
