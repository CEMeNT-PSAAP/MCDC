import math

from numba import njit, int64

import mcdc.mesh as mesh
import mcdc.physics as physics
import mcdc.adapt as adapt
import mcdc.type_ as type_

from mcdc.adapt import for_cpu, for_gpu
from mcdc.algorithm import binary_search
from mcdc.constant import *


# ======================================================================================
# Geometry inspection
# ======================================================================================


@njit
def inspect_geometry(particle_container, mcdc):
    """
    Full geometry inspection of the particle:
        - Set particle top cell and material IDs (if not lost)
        - Set surface ID (if surface hit)
        - Set particle boundary event (surface or lattice crossing, or lost)
        - Return distance to boundary (surface or lattice)
    """
    particle = particle_container[0]

    # Store particle global coordinate
    # (particle will be temporarily translated and rotated)
    x_global = particle["x"]
    y_global = particle["y"]
    z_global = particle["z"]
    t_global = particle["t"]
    ux_global = particle["ux"]
    uy_global = particle["uy"]
    uz_global = particle["uz"]
    speed = physics.get_speed(particle_container, mcdc)

    # Default returns
    distance = INF
    event = EVENT_NONE

    # Find top cell from root universe if unknown
    if particle["cell_ID"] == -1:
        particle["cell_ID"] = get_cell(particle_container, UNIVERSE_ROOT, mcdc)

        # Particle is lost?
        if particle["cell_ID"] == -1:
            event = EVENT_LOST

    # The top cell
    cell = mcdc["cells"][particle["cell_ID"]]

    # Recursively check cells until material cell is found (or the particle is lost)
    while event != EVENT_LOST:
        # Distance to nearest surface
        d_surface, surface_ID = distance_to_nearest_surface(
            particle_container, cell, mcdc
        )

        # Check if smaller
        if d_surface < distance - COINCIDENCE_TOLERANCE:
            distance = d_surface
            event = EVENT_SURFACE_CROSSING
            particle["surface_ID"] = surface_ID

        # Check if coincident
        elif check_coincidence(d_surface, distance):
            # Add event if not there yet
            if not event & EVENT_SURFACE_CROSSING:
                event += EVENT_SURFACE_CROSSING
                particle["surface_ID"] = surface_ID
            # If surface crossing is already there, prioritize the outer surface ID

        # Material cell?
        if cell["fill_type"] == FILL_MATERIAL:
            particle["material_ID"] = cell["fill_ID"]
            break

        else:
            # Cell is filled with universe or lattice

            # Apply translation
            if cell["fill_translated"]:
                particle["x"] -= cell["translation"][0]
                particle["y"] -= cell["translation"][1]
                particle["z"] -= cell["translation"][2]

            if cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = mcdc["lattices"][cell["fill_ID"]]

                # Distance to lattice grid
                d_lattice = mesh.uniform.get_crossing_distance(
                    particle_container, speed, lattice
                )

                # Check if smaller
                if d_lattice < distance - COINCIDENCE_TOLERANCE:
                    distance = d_lattice
                    event = EVENT_LATTICE_CROSSING
                    particle["surface_ID"] = -1

                # Check if coincident
                if check_coincidence(d_lattice, distance):
                    # Add event if not there yet
                    if not event & EVENT_LATTICE_CROSSING:
                        event += EVENT_LATTICE_CROSSING

                # Get universe
                ix, iy, iz, it, outside = mesh.uniform.get_indices(
                    particle_container, lattice
                )
                if outside:
                    event = EVENT_LOST
                    continue
                universe_ID = lattice["universe_IDs"][ix, iy, iz]

                # Lattice-translate the particle
                particle["x"] -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
                particle["y"] -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
                particle["z"] -= lattice["z0"] + (iz + 0.5) * lattice["dz"]

                # Get inner cell
                cell_ID = get_cell(particle_container, universe_ID, mcdc)
                if cell_ID > -1:
                    cell = mcdc["cells"][cell_ID]
                else:
                    event = EVENT_LOST

    # Reassign the global coordinate
    particle["x"] = x_global
    particle["y"] = y_global
    particle["z"] = z_global
    particle["t"] = t_global
    particle["ux"] = ux_global
    particle["uy"] = uy_global
    particle["uz"] = uz_global

    # Report lost particle
    if event == EVENT_LOST:
        report_lost(particle_container)

    # Assign particle event
    particle["event"] = event

    return distance


@njit
def locate_particle(particle_container, mcdc):
    """
    Set particle cell and material IDs
    Return False if particle is lost

    This is similar to inspect_geometry, except that distance to nearest surface
    or/and lattice grid and the respective boundary event are not determined.
    """
    particle = particle_container[0]

    # Store particle global coordinate
    # (particle will be temporarily translated and rotated)
    x_global = particle["x"]
    y_global = particle["y"]
    z_global = particle["z"]
    t_global = particle["t"]
    ux_global = particle["ux"]
    uy_global = particle["uy"]
    uz_global = particle["uz"]
    speed = physics.get_speed(particle_container, mcdc)

    particle_is_lost = False

    # Find top cell from root universe if unknown
    if particle["cell_ID"] == -1:
        particle["cell_ID"] = get_cell(particle_container, UNIVERSE_ROOT, mcdc)

        # Particle is lost?
        if particle["cell_ID"] == -1:
            particle_is_lost = True

    # The top cell
    cell = mcdc["cells"][particle["cell_ID"]]

    # Recursively check cells until material cell is found (or the particle is lost)
    while not particle_is_lost:
        # Material cell?
        if cell["fill_type"] == FILL_MATERIAL:
            particle["material_ID"] = cell["fill_ID"]
            break

        else:
            # Cell is filled with universe or lattice

            # Apply translation
            if cell["fill_translated"]:
                particle["x"] -= cell["translation"][0]
                particle["y"] -= cell["translation"][1]
                particle["z"] -= cell["translation"][2]

            if cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = mcdc["lattices"][cell["fill_ID"]]

                # Get universe
                ix, iy, iz, it, outside = mesh.uniform.get_indices(
                    particle_container, lattice
                )
                if outside:
                    particle_is_lost = True
                    continue
                universe_ID = lattice["universe_IDs"][ix, iy, iz]

                # Lattice-translate the particle
                particle["x"] -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
                particle["y"] -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
                particle["z"] -= lattice["z0"] + (iz + 0.5) * lattice["dz"]

                # Get inner cell
                cell_ID = get_cell(particle_container, universe_ID, mcdc)
                if cell_ID > -1:
                    cell = mcdc["cells"][cell_ID]
                else:
                    particle_is_lost = True

    # Reassign the global coordinate
    particle["x"] = x_global
    particle["y"] = y_global
    particle["z"] = z_global
    particle["t"] = t_global
    particle["ux"] = ux_global
    particle["uy"] = uy_global
    particle["uz"] = uz_global

    # Report lost particle
    if particle_is_lost:
        report_lost(particle_container)

    return not particle_is_lost


# ======================================================================================
# Particle locator
# ======================================================================================


@njit
def get_cell(particle_container, universe_ID, mcdc):
    """
    Find and return particle cell ID in the given universe
    Return -1 if particle is lost
    """
    particle = particle_container[0]
    universe = mcdc["universes"][universe_ID]

    # Check all cells in the universe
    for cell_ID in universe["cell_IDs"]:
        cell = mcdc["cells"][cell_ID]
        if check_cell(particle_container, cell, mcdc):
            return cell["ID"]

    # Particle is not found
    return -1


@njit
def check_cell(particle_container, cell, mcdc):
    """
    Check if the particle is inside the cell
    """
    particle = particle_container[0]

    # Access RPN data
    idx = cell["region_data_idx"]
    N_token = mcdc["cell_region_data"][idx]

    # Create local value array
    value = adapt.local_array(type_.rpn_buffer_size(), type_.bool_)
    N_value = 0

    # March forward through RPN tokens
    idx += 1
    idx_end = idx + N_token
    while idx < idx_end:
        token = mcdc["cell_region_data"][idx]

        if token >= 0:
            surface = mcdc["surfaces"][token]
            value[N_value] = check_surface_sense(particle_container, surface)
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
def check_surface_sense(particle_container, surface):
    """
    Check on which side of the surface the particle is
        - Return True if positive side
        - Return False otherwise
    Particle direction is used if coincide within the tolerance
    """
    result = surface_evaluate(particle_container, surface)

    # Check if coincident on the surface
    if abs(result) < COINCIDENCE_TOLERANCE:
        # Determine sense based on the direction
        return surface_normal_component(particle_container, surface) > 0.0

    return result > 0.0


@for_cpu()
def report_lost(particle_container):
    """
    Report lost particle and terminate it
    """
    particle = particle_container[0]

    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    print("A particle is lost at (", x, y, z, ")")
    particle["alive"] = False


@for_gpu()
def report_lost(particle_container):
    particle = particle_container[0]

    particle["alive"] = False


# ======================================================================================
# Nearest distance search
# ======================================================================================


@njit
def distance_to_nearest_surface(particle_container, cell, mcdc):
    """
    The termine the nearest cell surface and the distance to it
    """
    # TODO: docs
    particle = particle_container[0]
    distance = INF
    surface_ID = -1

    # Access cell surface data
    idx = cell["surface_data_idx"]
    N_surface = mcdc["cell_surface_data"][idx]

    # Iterate over all surfaces
    idx += 1
    idx_end = idx + N_surface
    while idx < idx_end:
        candidate_surface_ID = mcdc["cell_surface_data"][idx]
        surface = mcdc["surfaces"][candidate_surface_ID]
        d = surface_distance(particle_container, surface, mcdc)
        if d < distance:
            distance = d
            surface_ID = surface["ID"]
        idx += 1
    return distance, surface_ID


# =============================================================================
# Surface operations
# =============================================================================
# Quadric surface: Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
# TODO: make movement a translation and rotation


@njit
def surface_evaluate(particle_container, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    """
    particle = particle_container[0]
    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]

    # Surface coefficient
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    # Linear surface evaluation
    result = G * x + H * y + I * z + J
    if surface["linear"]:
        return result

    # Surface coefficient
    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]

    # Quadric surface evaluation
    return (
        result + A * x * x + B * y * y + C * z * z + D * x * y + E * x * z + F * y * z
    )


@njit
def surface_distance(particle_container, surface, mcdc):
    """
    Return particle distance to surface
    """
    particle = particle_container[0]
    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Check if coincident and leaving the surface
    evaluation = surface_evaluate(particle_container, surface)
    coincident = False
    if abs(evaluation) < COINCIDENCE_TOLERANCE:
        coincident = True
        if surface["linear"]:
            return INF
        elif surface_normal_component(particle_container, surface) > 0.0:
            return INF

    # Surface coefficients
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    # Distance to linear surface
    if surface["linear"]:
        distance = -evaluation / (G * ux + H * uy + I * uz)

        # Moving away from the surface
        if distance < 0.0:
            return INF
        else:
            return distance

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


@njit
def surface_bc(particle_container, surface):
    """
    Apply surface boundary condition to the particle
    """
    particle = particle_container[0]
    if surface["BC"] == BC_VACUUM:
        particle["alive"] = False
    elif surface["BC"] == BC_REFLECTIVE:
        surface_reflect(particle_container, surface)


@njit
def surface_reflect(particle_container, surface):
    """
    Surface-reflect the particle
    """
    particle = particle_container[0]
    # Particle coordinate
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    nx, ny, nz = surface_normal(particle_container, surface)

    c = 2.0 * (nx * ux + ny * uy + nz * uz)

    particle["ux"] = ux - c * nx
    particle["uy"] = uy - c * ny
    particle["uz"] = uz - c * nz


@njit
def surface_normal(particle_container, surface):
    """
    Get the surface outward-normal vector at the particle coordinate
    """
    particle = particle_container[0]
    if surface["linear"]:
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
def surface_normal_component(particle_container, surface):
    """
    Get the surface outward-normal component of the particle
    (dot product of the two directional vectors)
    """
    particle = particle_container[0]
    # Surface outward-normal vector
    nx, ny, nz = surface_normal(particle_container, surface)

    # Particle direction vector
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # The dot product
    return nx * ux + ny * uy + nz * uz


# ======================================================================================
# Miscellanies
# ======================================================================================


@njit
def check_coincidence(value_1, value_2):
    """
    Check if two values are within coincidence tolerance
    """
    return abs(value_1 - value_2) < COINCIDENCE_TOLERANCE
