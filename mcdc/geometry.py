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
    Full geometry inspection of the particle:
        - Set particle top cell and material IDs (if not lost)
        - Set surface ID (if surface hit)
        - Return distance to boundary (surface or lattice)
        - Return event type (surface or lattice hit or particle lost)
    """
    # TODO: add universe cell (besides material and lattice cells)

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
        particle["cell_ID"] = get_cell(x, y, z, t, ux, uy, uz, UNIVERSE_ROOT, mcdc)
        # Particle is lost?
        if particle["cell_ID"] == -1:
            lost(particle)
            return 0.0, EVENT_LOST

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

                # Get inner cell
                cell_ID = get_cell(x, y, z, t, ux, uy, uz, universe_ID, mcdc)
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
def get_cell(x, y, z, t, ux, uy, uz, universe_ID, mcdc):
    """
    Find and return cell ID of the given local coordinate in the given universe
    Return -1 if particle is lost
    """
    universe = mcdc["universes"][universe_ID]

    # Check all cells in the universe
    for cell_ID in universe["cell_IDs"]:
        cell = mcdc["cells"][cell_ID]
        if cell_check(x, y, z, t, ux, uy, uz, cell, mcdc):
            return cell["ID"]

    # Particle is not found
    return -1


@njit
def cell_check(x, y, z, t, ux, uy, uz, cell, mcdc):
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
            value[N_value] = surface_sense_check(x, y, z, t, ux, uy, uz, surface)
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
def surface_sense_check(x, y, z, t, ux, uy, uz, surface):
    """
    Check on which side of the surface the local coordinate is
        - Return True if positive side
        - Return False otherwise
    The given local direction is used if coincide within the tolerance
    """
    result = surface_evaluate(x, y, z, t, surface)

    # Check if coincident on the surface
    if abs(result) < COINCIDENCE_TOLERANCE:
        # Determine sense based on the direction
        return surface_normal_component(x, y, z, t, ux, uy, uz, surface) > 0.0

    return result > 0.0


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
    # TODO: docs
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
    # TODO: docs
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
    # TODO: docs
    ix = int64(math.floor((x - lattice["x0"]) / lattice["dx"]))
    iy = int64(math.floor((y - lattice["y0"]) / lattice["dy"]))
    iz = int64(math.floor((z - lattice["z0"]) / lattice["dz"]))
    return ix, iy, iz


# =============================================================================
# Surface operations
# =============================================================================
# Quadric surface: Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
# TODO: make movement a translation and rotation


@njit
def surface_evaluate(x, y, z, t, surface):
    # TODO: docs
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
    # TODO: docs

    # Check if coincident
    evaluation = surface_evaluate(x, y, z, t, surface)
    if abs(evaluation) < COINCIDENCE_TOLERANCE:
        return INF, False

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
        distance = -evaluation / (
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
    c = evaluation

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
    # TODO: docs
    if surface["BC"] == BC_VACUUM:
        particle["alive"] = False
    elif surface["BC"] == BC_REFLECTIVE:
        surface_reflect(particle, surface)


@njit
def surface_reflect(particle, surface):
    # TODO: docs
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    nx, ny, nz = surface_normal(x, y, z, t, surface)
    # 2.0*surface_normal_component(...)
    c = 2.0 * (nx * ux + ny * uy + nz * uz)

    particle["ux"] = ux - c * nx
    particle["uy"] = uy - c * ny
    particle["uz"] = uz - c * nz


@njit
def surface_normal(x, y, z, t, surface):
    # TODO: docs
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

    dx = 2 * A * x + D * y + E * z + G
    dy = 2 * B * y + D * x + F * z + H
    dz = 2 * C * z + E * x + F * y + I_

    norm = (dx**2 + dy**2 + dz**2) ** 0.5
    return dx / norm, dy / norm, dz / norm


@njit
def surface_normal_component(x, y, z, t, ux, uy, uz, surface):
    # TODO: docs
    nx, ny, nz = surface_normal(x, y, z, t, surface)
    return nx * ux + ny * uy + nz * uz
