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
def reflect(particle, surface):
    """
    Reflect the particle off the surface
    """
    # Particle coordinate
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    nx, ny, nz = get_normal(particle, surface)

    c = 2.0 * (nx * ux + ny * uy + nz * uz)

    particle["ux"] = ux - c * nx
    particle["uy"] = uy - c * ny
    particle["uz"] = uz - c * nz


@njit
def get_normal(particle, surface):
    """
    Get the surface outward-normal vector at the particle coordinate
    """
    if surface["type"] & SURFACE_LINEAR:
        return surface["nx"], surface["ny"], surface["nz"]

    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]

    # Surface coefficients
    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]

    dx = 2 * A * x + D * y + E * z + G
    dy = 2 * B * y + D * x + F * z + H
    dz = 2 * C * z + E * x + F * y + I

    norm = (dx**2 + dy**2 + dz**2) ** 0.5
    return dx / norm, dy / norm, dz / norm


@njit
def get_normal_component(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle
    (dot product of the two directional vectors)
    """
    # Surface outward-normal vector
    nx, ny, nz = get_normal(particle, surface)

    # Particle direction vector
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # The dot product
    if not surface["moving"]:
        return nx * ux + ny * uy + nz * uz
    else:
        # Get move index
        N_move = surface["N_move"]
        time_grid = surface["move_time_grid"]
        idx = binary_search_with_length(particle["t"], time_grid, N_move)
        # Coinciding cases
        if abs(time_grid[idx + 1] - particle["t"]) < COINCIDENCE_TOLERANCE:
            idx += 1

        # The relative velocity
        translation_start = surface["move_translations"][idx]
        translation_end = surface["move_translations"][idx + 1]
        dx = translation_end[0] - translation_start[0]
        dy = translation_end[1] - translation_start[1]
        dz = translation_end[2] - translation_start[2]
        dt = time_grid[idx + 1] - time_grid[idx]
        relative_velocity = (nx * ux + ny * uy + nz * uz) - (
            nx * dx + ny * dy + nz * dz
        ) / dt / speed

        return relative_velocity


@njit
def get_distance(particle, speed, surface):
    """
    Get particle distance to surface
    """
    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Check if coincident and leaving the surface forever
    evaluation = evaluate(particle, surface)
    coincident = abs(evaluation) < COINCIDENCE_TOLERANCE
    if coincident:
        if surface["type"] & SURFACE_LINEAR:
            if not surface["moving"]:
                return INF
        elif get_normal_component(particle, speed, surface) > 0.0:
            return INF

    # Surface coefficients
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    # Distance to linear surface
    if surface["type"] & SURFACE_LINEAR:
        if not surface["moving"]:
            distance = -evaluation / (G * ux + H * uy + I * uz)

            # Moving away from the surface
            if distance < 0.0:
                return INF
            else:
                return distance

        # The surface is moving

        # Get starting interval index
        N_move = surface["N_move"]
        time_grid = surface["move_time_grid"]
        idx = binary_search_with_length(t, time_grid, N_move)
        # Coinciding cases
        if abs(time_grid[idx + 1] - t) < COINCIDENCE_TOLERANCE:
            idx += 1

        # ==============================================================================
        # Evaluate starting interval if not coincident

        if not coincident:
            # The relative velocity
            translation_start = surface["move_translations"][idx]
            translation_end = surface["move_translations"][idx + 1]
            dx = translation_end[0] - translation_start[0]
            dy = translation_end[1] - translation_start[1]
            dz = translation_end[2] - translation_start[2]
            dt = time_grid[idx + 1] - time_grid[idx]
            relative_velocity = (G * ux + H * uy + I * uz) - (
                G * dx + H * dy + I * dz
            ) / dt / speed

            # Check if particle and surface move in the same direction
            if relative_velocity != 0.0:
                distance = -evaluation / relative_velocity
                # Check if not moving away
                if distance > 0.0:
                    # Check if it is still within interval
                    distance_time = distance / speed
                    if distance_time <= time_grid[idx + 1] - t:
                        return distance

        # Not intersecting in the starting interval. Let's check the next ones
        idx += 1

        # But first, we need to keep track of the total distance traveled and
        # the particle position (which we temporarily change)
        total_distance = (time_grid[idx] - t) * speed
        particle["x"] += total_distance * ux
        particle["y"] += total_distance * uy
        particle["z"] += total_distance * uz

        # ==============================================================================
        # Evaluate subsequent interval

        while idx < N_move:
            # The relative velocity
            translation_start = surface["move_translations"][idx]
            translation_end = surface["move_translations"][idx + 1]
            dx = translation_end[0] - translation_start[0]
            dy = translation_end[1] - translation_start[1]
            dz = translation_end[2] - translation_start[2]
            dt = time_grid[idx + 1] - time_grid[idx]
            relative_velocity = (G * ux + H * uy + I * uz) - (
                G * dx + H * dy + I * dz
            ) / dt / speed

            # Check if particle and surface move in the same direction
            if relative_velocity != 0.0:
                distance = -evaluate(particle, surface) / relative_velocity
                # Check if not moving away
                if distance > 0.0:
                    # Check if it is still within interval
                    distance_time = distance / speed
                    if distance_time <= dt:
                        # Restore particle coordinate
                        particle["x"] = x
                        particle["y"] = y
                        particle["z"] = z

                        return total_distance + distance

            # Not intersecting within the current interval
            additional_distance = dt * speed
            total_distance += additional_distance
            particle["x"] += additional_distance * ux
            particle["y"] += additional_distance * uy
            particle["z"] += additional_distance * uz

            idx += 1

        # No intersection
        distance = INF

        # Restore particle coordinate
        particle["x"] = x
        particle["y"] = y
        particle["z"] = z

        return INF

    # Surface coefficients
    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]

    # Quadratic equation constants
    a = (
        A * ux * ux
        + B * uy * uy
        + C * uz * uz
        + D * ux * uy
        + E * ux * uz
        + F * uy * uz
    )
    b = (
        2 * (A * x * ux + B * y * uy + C * z * uz)
        + D * (x * uy + y * ux)
        + E * (x * uz + z * ux)
        + F * (y * uz + z * uy)
        + G * ux
        + H * uy
        + I * uz
    )
    c = evaluation

    determinant = b * b - 4.0 * a * c

    # Roots are complex  : no intersection
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

        # Coincident treatment
        if coincident:
            return max(root_1, root_2)

        # Negative roots, moving away from the surface
        if root_1 < 0.0:
            root_1 = INF
        if root_2 < 0.0:
            root_2 = INF

        # Return the smaller root
        return min(root_1, root_2)


# ======================================================================================
# Prrivate
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
