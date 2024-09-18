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
from mcdc.src.algorithm import binary_search_with_length


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
    if surface["moving"]:
        # Temporarily translate particle position
        x_original = particle["x"]
        y_original = particle["y"]
        z_original = particle["z"]
        idx = _get_move_idx(particle["t"], surface)
        _translate_particle_position(particle, surface, idx)

    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            result = plane_x.evaluate(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            result = plane_y.evaluate(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            result = plane_z.evaluate(particle, surface)
        elif surface["type"] & SURFACE_PLANE:
            result = plane.evaluate(particle, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            result = cylinder_x.evaluate(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            result = cylinder_y.evaluate(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            result = cylinder_z.evaluate(particle, surface)
        elif surface["type"] & SURFACE_SPHERE:
            result = sphere.evaluate(particle, surface)
        else:
            result = quadric.evaluate(particle, surface)

    if surface["moving"]:
        # Restore particle position
        particle["x"] = x_original
        particle["y"] = y_original
        particle["z"] = z_original

    return result


@njit
def get_normal_component(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle
    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative direction.
    """
    if surface["moving"]:
        # Temporarily translate particle direction
        ux_original = particle["ux"]
        uy_original = particle["uy"]
        uz_original = particle["uz"]
        idx = _get_move_idx(particle["t"], surface)
        _translate_particle_direction(particle, speed, surface, idx)

    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            result = plane_x.get_normal_component(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            result = plane_y.get_normal_component(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            result = plane_z.get_normal_component(particle, surface)
        elif surface["type"] & SURFACE_PLANE:
            result = plane.get_normal_component(particle, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            result = cylinder_x.get_normal_component(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            result = cylinder_y.get_normal_component(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            result = cylinder_z.get_normal_component(particle, surface)
        elif surface["type"] & SURFACE_SPHERE:
            result = sphere.get_normal_component(particle, surface)
        else:
            result = quadric.get_normal_component(particle, surface)

    if surface["moving"]:
        # Restore particle direction
        particle["ux"] = ux_original
        particle["uy"] = uy_original
        particle["uz"] = uz_original

    return result


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

    Particle speed is needed if the surface is moving.
    """
    if surface["moving"]:
        return _get_distance_moving(particle, speed, surface)
    else:
        return _get_distance_static(particle, surface)


@njit
def _get_distance_static(particle, surface):
    """
    Get particle distance to static surface
    """
    if surface["type"] & SURFACE_LINEAR:
        if surface["type"] & SURFACE_PLANE_X:
            return plane_x.get_distance(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Y:
            return plane_y.get_distance(particle, surface)
        elif surface["type"] & SURFACE_PLANE_Z:
            return plane_z.get_distance(particle, surface)
        elif surface["type"] & SURFACE_PLANE:
            return plane.get_distance(particle, surface)
    else:
        if surface["type"] & SURFACE_CYLINDER_X:
            return cylinder_x.get_distance(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Y:
            return cylinder_y.get_distance(particle, surface)
        elif surface["type"] & SURFACE_CYLINDER_Z:
            return cylinder_z.get_distance(particle, surface)
        elif surface["type"] & SURFACE_SPHERE:
            return sphere.get_distance(particle, surface)
        else:
            return quadric.get_distance(particle, surface)


@njit
def _get_distance_moving(particle, speed, surface):
    """
    Get particle distance to moving surface
    """
    # Store original particle and surface parameters (will be temporarily changed)
    x_original = particle["x"]
    y_original = particle["y"]
    z_original = particle["z"]
    ux_original = particle["ux"]
    uy_original = particle["uy"]
    uz_original = particle["uz"]
    t_original = particle["t"]
    surface["moving"] = False

    # Move interval index
    idx = _get_move_idx(particle["t"], surface)

    # Distance accumulator
    total_distance = 0.0

    # Evaluate the current and the subsequent intervals until intersecting
    while idx < surface["N_move"]:
        # Translate particle position and direction
        _translate_particle_position(particle, surface, idx)
        _translate_particle_direction(particle, speed, surface, idx)

        # Get distance
        distance = _get_distance_static(particle, surface)

        # Beyond the interval?
        distance_time = distance / speed
        dt = surface["move_time_grid"][idx + 1] - particle["t"]
        if distance_time > dt:
            distance = INF

        # Intersecting?
        if distance < INF:
            # Restore particle and surface parameters
            particle["x"] = x_original
            particle["y"] = y_original
            particle["z"] = z_original
            particle["ux"] = ux_original
            particle["uy"] = uy_original
            particle["uz"] = uz_original
            particle["t"] = t_original
            surface["moving"] = True

            # Return the total distance
            return total_distance + distance

        # Accumulate distance
        additional_distance = dt * speed
        total_distance += additional_distance

        # Modify the particle
        particle["x"] = x_original + additional_distance * ux_original
        particle["y"] = y_original + additional_distance * uy_original
        particle["z"] = z_original + additional_distance * uz_original
        particle["ux"] = ux_original
        particle["uy"] = uy_original
        particle["uz"] = uz_original
        particle["t"] = surface["move_time_grid"][idx + 1]

        # Check next interval
        idx += 1

    # Restore particle and surface parameters
    particle["x"] = x_original
    particle["y"] = y_original
    particle["z"] = z_original
    particle["ux"] = ux_original
    particle["uy"] = uy_original
    particle["uz"] = uz_original
    particle["t"] = t_original
    surface["moving"] = True

    # No intersection
    return INF


# ======================================================================================
# Private
# ======================================================================================


@njit
def _get_move_idx(t, surface):
    """
    Get moving interval index wrt the given time
    """
    N_move = surface["N_move"]
    time_grid = surface["move_time_grid"]
    idx = binary_search_with_length(t, time_grid, N_move)

    # Coinciding cases
    if abs(time_grid[idx + 1] - t) < COINCIDENCE_TOLERANCE:
        idx += 1

    return idx


@njit
def _translate_particle_position(particle, surface, idx):
    """
    Translate particle position wrt the given surface moving interval index
    """

    # Surface move translations, velocities, and time grid
    trans_0 = surface["move_translations"][idx]
    time_0 = surface["move_time_grid"][idx]
    V = surface["move_velocities"][idx]

    # Translate the particle
    t_local = particle["t"] - time_0
    particle["x"] -= trans_0[0] + V[0] * t_local
    particle["y"] -= trans_0[1] + V[1] * t_local
    particle["z"] -= trans_0[2] + V[2] * t_local


@njit
def _translate_particle_direction(particle, speed, surface, idx):
    """
    Translate particle direction wrt the given surface moving interval index
    """

    # Surface move translations, velocities, and time grid
    V = surface["move_velocities"][idx]

    # Translate the particle
    particle["ux"] -= V[0] / speed
    particle["uy"] -= V[1] / speed
    particle["uz"] -= V[2] / speed
