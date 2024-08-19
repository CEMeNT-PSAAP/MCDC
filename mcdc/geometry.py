import math

from numba import njit, int64

import mcdc.local as local
import mcdc.physics as physics

from mcdc.algorithm import binary_search
from mcdc.constant import *


# ======================================================================================
# Geometry inspection
# ======================================================================================


@njit
def inspect_geometry(particle, mcdc):
    """
    Full geometry inspection of the particle
    This function
        - sets particle top cell and material IDs (if not lost)
        - sets surface ID (if surface hit)
        - returns distance to boundary (surface or lattice)
        - returns event type (surface or lattice hit or particle lost)
    """
    # TODO: add universe cell, besides lattice and material cells

    # Particle local coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    v = physics.get_speed(particle, mcdc)

    # Default returns
    distance = INF
    event = EVENT_LOST

    # Find top cell from root universe if unknown
    if particle["cell_ID"] == -1:
        particle["cell_ID"] = get_cell(x, y, z, t, UNIVERSE_ROOT, mcdc)
        # Particle is lost?
        if particle["cell_ID"] == -1:
            lost(particle)
            return 0.0, EVENT_LOST

    # TODO: temporary
    particle["translation"][0] = 0.0
    particle["translation"][1] = 0.0
    particle["translation"][2] = 0.0

    # The top cell
    cell = mcdc["cells"][particle["cell_ID"]]

    # Recursively check cell until material cell is found
    while True:
        # Distance to nearest surface
        d_surface, surface_ID, surface_move = distance_to_nearest_surface(
            x, y, z, t, ux, uy, uz, v, cell, mcdc
        )

        # Check if smaller
        if d_surface * PREC < distance:
            distance = d_surface
            event = EVENT_SURFACE
            particle["surface_ID"] = surface_ID

            if surface_move:
                event = EVENT_SURFACE_MOVE

        # Material cell?
        if cell["fill_type"] == FILL_MATERIAL:
            particle["material_ID"] = cell["fill_ID"]
            break

        else:
            # Cell is filled with universe or lattice

            # Apply translation
            if cell["fill_translated"]:
                x -= cell["translation"][0]
                y -= cell["translation"][1]
                z -= cell["translation"][2]

                particle["translation"][0] += cell["translation"][0]
                particle["translation"][1] += cell["translation"][1]
                particle["translation"][2] += cell["translation"][2]

            if cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = mcdc["lattices"][cell["fill_ID"]]

                # Distance to lattice grid
                d_lattice = distance_to_lattice_grid(x, y, z, ux, uy, uz, lattice)

                # Check if smaller
                if d_lattice * PREC < distance:
                    distance = d_lattice
                    event = EVENT_LATTICE
                    particle["surface_ID"] = -1

                # Get universe
                ix, iy, iz = lattice_get_index(x, y, z, lattice)
                universe_ID = lattice["universe_IDs"][ix, iy, iz]

                # Lattice-translate the particle
                x -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
                y -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
                z -= lattice["z0"] + (iz + 0.5) * lattice["dz"]

                particle["translation"][0] += lattice["x0"] + (ix + 0.5) * lattice["dx"]
                particle["translation"][1] += lattice["y0"] + (iy + 0.5) * lattice["dy"]
                particle["translation"][2] += lattice["z0"] + (iz + 0.5) * lattice["dz"]

                # Get inner cell
                cell_ID = get_cell(x, y, z, t, universe_ID, mcdc)
                if cell_ID > -1:
                    cell = mcdc["cells"][cell_ID]
                else:
                    # Skip if particle is lost
                    return 0.0, EVENT_LOST

    return distance, event


# ======================================================================================
# Particle locator
# ======================================================================================


@njit
def get_cell(x, y, z, t, universe_ID, mcdc):
    """
    Find and return cell ID of the given local coordinate in the universe
    Return -1 if particle is lost
    """
    universe = mcdc["universes"][universe_ID]

    # Check all cells in the universe
    for cell_ID in universe["cell_IDs"]:
        cell = mcdc["cells"][cell_ID]
        if cell_check(x, y, z, t, cell, mcdc):
            return cell["ID"]

    # Particle is not found
    return -1


@njit
def cell_check(x, y, z, t, cell, mcdc):
    """
    Check if the given local coordinate is inside the cell
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
            value[N_value] = surface_evaluate(x, y, z, t, surface) > 0.0
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


@njit
def lost(particle):
    """
    Report lost particle and terminate it
    """
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    print("A particle is lost at (", x, y, z, ")")
    particle["alive"] = False


# ======================================================================================
# Nearest distance search
# ======================================================================================


@njit
def distance_to_nearest_surface(x, y, z, t, ux, uy, uz, v, cell, mcdc):
    # TODO: docs
    distance = INF
    surface_ID = -1
    surface_move = False

    # Access cell surface data
    idx = cell["surface_data_idx"]
    N_surface = mcdc["cell_surface_data"][idx]

    # Iterate over all surfaces
    idx += 1
    idx_end = idx + N_surface
    while idx < idx_end:
        candidate_surface_ID = mcdc["cell_surface_data"][idx]
        surface = mcdc["surfaces"][candidate_surface_ID]
        d, sm = surface_distance(x, y, z, t, ux, uy, uz, v, surface, mcdc)
        if d < distance:
            distance = d
            surface_ID = surface["ID"]
            surface_move = sm
        idx += 1
    return distance, surface_ID, surface_move


@njit
def distance_to_lattice_grid(x, y, z, ux, uy, uz, lattice):
    d = INF
    d = min(d, lattice_grid_distance(x, ux, lattice["x0"], lattice["dx"]))
    d = min(d, lattice_grid_distance(y, uy, lattice["y0"], lattice["dy"]))
    d = min(d, lattice_grid_distance(z, uz, lattice["z0"], lattice["dz"]))
    return d


# ======================================================================================
# Lattice operations
# ======================================================================================


@njit
def lattice_grid_distance(value, direction, x0, dx):
    if direction == 0.0:
        return INF
    idx = math.floor((value - x0) / dx)
    if direction > 0.0:
        idx += 1
    ref = x0 + idx * dx
    dist = (ref - value) / direction
    return dist


@njit
def lattice_get_index(x, y, z, lattice):
    ix = int64(math.floor((x - lattice["x0"]) / lattice["dx"]))
    iy = int64(math.floor((y - lattice["y0"]) / lattice["dy"]))
    iz = int64(math.floor((z - lattice["z0"]) / lattice["dz"]))
    return ix, iy, iz


# =============================================================================
# Surface operations
# =============================================================================
# Quadric surface: Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
# TODO: replace shifting mechanics with intersection tolerance
# TODO: make movement a translation and rotation


@njit
def surface_evaluate(x, y, z, t, surface):
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
def surface_distance(x, y, z, t, ux, uy, uz, v, surface, mcdc):
    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    surface_move = False
    if surface["linear"]:
        idx = 0
        if surface["N_slice"] > 1:
            idx = binary_search(t, surface["t"][: surface["N_slice"] + 1])
        J1 = surface["J"][idx][1]

        t_max = surface["t"][idx + 1]
        d_max = (t_max - t) * v

        div = G * ux + H * uy + I_ * uz + J1 / v
        distance = -surface_evaluate(x, y, z, t, surface) / (
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
    c = surface_evaluate(x, y, z, t, surface)

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


@njit
def surface_bc(particle, surface):
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

    # TODO: temporary
    x = particle["x"] - particle["translation"][0]
    y = particle["y"] - particle["translation"][1]
    z = particle["z"] - particle["translation"][2]

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
