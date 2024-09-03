"""
Plane: General linear surface

f(x, y, z) = Gx + Hy + Iz + J
"""

from numba import njit

import mcdc.src.surface.common as common

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)


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
    Evaluate the surface equation wrt the particle coordinate [Moving version]

    f(x(t), y(t), z(t)) = G * x(t) + H * y(t) + I * z(t) + J,
    and
        x(t) = x0 - Vx * t,
        y(t) = y0 - Vy * t,
        z(t) = z0 - Vz * t,
    where V is the move translation velocity.
    """
    # Surface parameters
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Translation velocity
    Vx = surface["move_velocities"][idx][0]
    Vy = surface["move_velocities"][idx][1]
    Vz = surface["move_velocities"][idx][2]

    # Translated position
    t_local = particle["t"] - surface["move_time_grid"][idx]
    x_translated = particle["x"] - Vx * t_local
    y_translated = particle["y"] - Vy * t_local
    z_translated = particle["z"] - Vz * t_local

    return G * x_translated + H * y_translated + I * z_translated + J


@njit
def reflect(particle, surface):
    """
    Reflect the particle off the surface
    """
    # Particle parameters
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Surface normal
    nx = surface["G"]
    ny = surface["H"]
    nz = surface["I"]

    # Reflect
    c = 2.0 * (nx * ux + ny * uy + nz * uz)
    particle["ux"] -= c * nx
    particle["uy"] -= c * ny
    particle["uz"] -= c * nz


@njit
def get_normal_component(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle

    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative direction.
    """
    if surface["moving"]:
        return get_normal_component_moving(particle, speed, surface)

    # Surface normal
    nx = surface["G"]
    ny = surface["H"]
    nz = surface["I"]

    # Particle parameters
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    return nx * ux + ny * uy + nz * uz


@njit
def get_normal_component_moving(particle, speed, surface):
    """
    Get the surface outward-normal component of the particle [Moving version]

    This is the dot product of the particle and the surface outward-normal directions.
    Particle speed is needed if the surface is moving to get the relative direction.
    """
    # Surface normal
    nx = surface["G"]
    ny = surface["H"]
    nz = surface["I"]

    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Translation velocity
    Vx = surface["move_velocities"][idx][0]
    Vy = surface["move_velocities"][idx][1]
    Vz = surface["move_velocities"][idx][2]

    # Relative direction
    rV_x = particle["ux"] - Vx / speed
    rV_y = particle["uy"] - Vy / speed
    rV_z = particle["uz"] - Vz / speed

    return nx * rVx + ny * rVy + nz * rVz


@njit
def get_distance(particle, speed, surface):
    """
    Get particle distance to surface
    """
    if surface["moving"]:
        return get_distance_moving(particle, speed, surface)

    # Parallel?
    normal_component = get_normal_component(particle, speed, surface)
    if abs(normal_component) == 0.0:
        return INF

    # Coincident?
    f = evaluate(particle, surface)
    if abs(f) < COINCIDENCE_TOLERANCE:
        return INF

    # Calculate distance
    distance = -f / normal_component

    # Moving away?
    if distance < 0.0:
        return INF

    return distance


@njit
def get_distance_moving(particle, speed, surface):
    """
    Get particle distance to surface [Moving version]
    """
    # Store particle coordinate (will be temporarily changed)
    x_original = particle["x"]
    y_original = particle["y"]
    z_original = particle["z"]
    ux_original = particle["ux"]
    uy_original = particle["uy"]
    uz_original = particle["uz"]
    t_original = particle["t"]
    surface["moving"] = False

    # Move interval index
    idx = common._get_move_idx(particle["t"], surface)

    # Distance accumulator
    total_distance = 0.0

    # Evaluate the current and the subsequent intervals until intersecting
    while idx < surface["N_move"]:
        # Apply translation velocity
        Vx = surface["move_velocities"][idx][0]
        Vy = surface["move_velocities"][idx][1]
        Vz = surface["move_velocities"][idx][2]
        particle["ux"] -= Vx / speed
        particle["uy"] -= Vy / speed
        particle["uz"] -= Vz / speed

        # Get distance
        distance = get_distance(particle, speed, surface)

        # Beyond the interval?
        distance_time = distance / speed
        dt = surface["move_time_grid"][idx + 1] - particle["t"]
        if distance_time > dt:
            distance = INF

        # Intersecting?
        if distance < INF:
            # Restore particle coordinate
            particle["x"] = x_original
            particle["y"] = y_original
            particle["z"] = z_original
            particle["ux"] = ux_original
            particle["uy"] = uy_original
            particle["uz"] = uz_original
            particle["t"] = t_original
            surface["moving"] = False

            # Return the total distance
            return total_distance + distance

        # Accumulate distance
        additional_distance = dt * speed
        total_distance += additional_distance

        # Modify the particle
        particle["x"] += additional_distance * ux_original
        particle["y"] += additional_distance * uy_original
        particle["z"] += additional_distance * uz_original
        particle["ux"] = ux_original
        particle["uy"] = uy_original
        particle["uz"] = uz_original
        particle["t"] = surface["move_time_grid"][idx + 1]

        # Check next interval
        idx += 1

    # Restore particle coordinate
    particle["x"] = x_original
    particle["y"] = y_original
    particle["z"] = z_original
    particle["ux"] = ux_original
    particle["uy"] = uy_original
    particle["uz"] = uz_original
    particle["t"] = t_original
    surface["moving"] = False

    # No intersection
    return INF
