import math

from numba import njit

import mcdc.local as local
import mcdc.physics as physics

from mcdc.algorithm import binary_search
from mcdc.constant import (
    BC_VACUUM,
    BC_REFLECTIVE,
    BOOL_AND,
    BOOL_NOT,
    BOOL_OR,
    INF,
    SHIFT,
)


# ======================================================================================
# Particle local coordinate
# ======================================================================================


@njit
def reset_local_coordinate(particle):
    """
    Reset particle's local coordinate
    """
    particle["translation"][0] = 0.0
    particle["translation"][1] = 0.0
    particle["translation"][2] = 0.0
    particle["translated"] = False


@njit
def get_local_coordinate(particle):
    """
    Get particle's local coordinate
    """
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]

    if particle["translated"]:
        x -= particle["translation"][0]
        y -= particle["translation"][1]
        z -= particle["translation"][2]

    return x, y, z


# ======================================================================================
# Particle locator
# ======================================================================================


@njit
def get_cell(particle, universe_ID, mcdc):
    """
    Find and return particle cell ID in the given universe
    """
    universe = mcdc["universes"][universe_ID]
    for cell_ID in universe["cell_IDs"]:
        cell = mcdc["cells"][cell_ID]
        if cell_check(particle, cell, mcdc):
            return cell["ID"]

    # Particle is not found
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    print("A particle is lost at (", x, y, z, ")")
    particle["alive"] = False
    return -1


@njit
def cell_check(particle, cell, mcdc):
    """
    Check if particle is inside the given cell
    """
    # Access RPN data
    idx = cell["region_data_idx"]
    N_token = mcdc["cell_region_data"][idx]

    # Create local value array
    value_struct = local.RPN_array()
    value = value_struct["values"]
    N_value = 0

    # March forward through RPN tokens
    idx += 1
    idx_end = idx + N_token
    while idx < idx_end:
        token = mcdc["cell_region_data"][idx]

        if token >= 0:
            surface = mcdc["surfaces"][token]
            value[N_value] = surface_evaluate(particle, surface) > 0.0
            N_value += 1

        elif token == BOOL_NOT:
            value[N_value - 1] = not value[N_value - 1]

        elif token == BOOL_AND:
            value[N_value - 2] = value[N_value - 2] & value[N_value - 1]
            N_value -= 1

        elif token == BOOL_OR:
            value[N_value - 2] = value[N_value - 2] | value[N_value - 1]
            N_value -= 1

        idx += 1

    return value[0]


# =============================================================================
# Surface operations
# =============================================================================
# Quadric surface: Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
# TODO: replace shifting mechanics with intersection tolerance
# TODO: make movement a translation and rotation


@njit
def surface_evaluate(particle, surface):
    x, y, z = get_local_coordinate(particle)
    t = particle["t"]

    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    # Get time indices
    idx = 0
    if surface["N_slice"] > 1:
        idx = binary_search(t, surface["t"][: surface["N_slice"] + 1])

    # Get constant
    J0 = surface["J"][idx][0]
    J1 = surface["J"][idx][1]
    J = J0 + J1 * (t - surface["t"][idx])

    result = G * x + H * y + I_ * z + J

    if surface["linear"]:
        return result

    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]

    return (
        result + A * x * x + B * y * y + C * z * z + D * x * y + E * x * z + F * y * z
    )


@njit
def apply_surface_bc(particle, surface):
    if surface["BC"] == BC_VACUUM:
        particle["alive"] = False
    elif surface["BC"] == BC_REFLECTIVE:
        surface_reflect(particle, surface)


@njit
def surface_reflect(particle, surface):
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    nx, ny, nz = surface_normal(particle, surface)
    # 2.0*surface_normal_component(...)
    c = 2.0 * (nx * ux + ny * uy + nz * uz)

    particle["ux"] = ux - c * nx
    particle["uy"] = uy - c * ny
    particle["uz"] = uz - c * nz


@njit
def surface_shift(particle, surface, mcdc):
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Get surface normal
    nx, ny, nz = surface_normal(particle, surface)

    # The shift
    shift_x = nx * SHIFT
    shift_y = ny * SHIFT
    shift_z = nz * SHIFT

    # Get dot product to determine shift sign
    if surface["linear"]:
        # Get time indices
        idx = 0
        if surface["N_slice"] > 1:
            idx = binary_search(particle["t"], surface["t"][: surface["N_slice"] + 1])
        J1 = surface["J"][idx][1]
        v = physics.get_speed(particle, mcdc)
        dot = ux * nx + uy * ny + uz * nz + J1 / v
    else:
        dot = ux * nx + uy * ny + uz * nz
    if dot > 0.0:
        particle["x"] += shift_x
        particle["y"] += shift_y
        particle["z"] += shift_z
    else:
        particle["x"] -= shift_x
        particle["y"] -= shift_y
        particle["z"] -= shift_z


@njit
def surface_normal(particle, surface):
    if surface["linear"]:
        return surface["nx"], surface["ny"], surface["nz"]

    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]
    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    x, y, z = get_local_coordinate(particle)

    dx = 2 * A * x + D * y + E * z + G
    dy = 2 * B * y + D * x + F * z + H
    dz = 2 * C * z + E * x + F * y + I_

    norm = (dx**2 + dy**2 + dz**2) ** 0.5
    return dx / norm, dy / norm, dz / norm


@njit
def surface_normal_component(particle, surface):
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    nx, ny, nz = surface_normal(particle, surface)
    return nx * ux + ny * uy + nz * uz


@njit
def surface_distance(particle, surface, mcdc):
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    surface_move = False
    if surface["linear"]:
        idx = 0
        if surface["N_slice"] > 1:
            idx = binary_search(particle["t"], surface["t"][: surface["N_slice"] + 1])
        J1 = surface["J"][idx][1]
        v = physics.get_speed(particle, mcdc)

        t_max = surface["t"][idx + 1]
        d_max = (t_max - particle["t"]) * v

        div = G * ux + H * uy + I_ * uz + J1 / v
        distance = -surface_evaluate(particle, surface) / (
            G * ux + H * uy + I_ * uz + J1 / v
        )

        # Go beyond current movement slice?
        if distance > d_max:
            distance = d_max
            surface_move = True
        elif distance < 0 and idx < surface["N_slice"] - 1:
            distance = d_max
            surface_move = True

        # Moving away from the surface
        if distance < 0.0:
            return INF, surface_move
        else:
            return distance, surface_move

    x, y, z = get_local_coordinate(particle)

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
        + I_ * uz
    )
    c = surface_evaluate(particle, surface)

    determinant = b * b - 4.0 * a * c

    # Roots are complex  : no intersection
    # Roots are identical: tangent
    # ==> return huge number
    if determinant <= 0.0:
        return INF, surface_move
    else:
        # Get the roots
        denom = 2.0 * a
        sqrt = math.sqrt(determinant)
        root_1 = (-b + sqrt) / denom
        root_2 = (-b - sqrt) / denom

        # Negative roots, moving away from the surface
        if root_1 < 0.0:
            root_1 = INF
        if root_2 < 0.0:
            root_2 = INF

        # Return the smaller root
        return min(root_1, root_2), surface_move
