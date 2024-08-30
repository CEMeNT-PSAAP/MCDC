import math

from numba import njit, int64

import mcdc.local as local
import mcdc.mesh as mesh
import mcdc.physics as physics

from mcdc.algorithm import binary_search_with_length
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
        - Set particle boundary event (surface or lattice crossing, or lost)
        - Return distance to boundary (surface or lattice)
    """
    # TODO: add universe cell (besides material and lattice cells)

    # Store particle global coordinate
    # (particle will be temporarily translated and rotated)
    x_global = particle["x"]
    y_global = particle["y"]
    z_global = particle["z"]
    t_global = particle["t"]
    ux_global = particle["ux"]
    uy_global = particle["uy"]
    uz_global = particle["uz"]
    speed = physics.get_speed(particle, mcdc)

    # Default returns
    distance = INF
    event = EVENT_NONE

    # Find top cell from root universe if unknown
    if particle["cell_ID"] == -1:
        particle["cell_ID"] = get_cell(particle, UNIVERSE_ROOT, mcdc)

        # Particle is lost?
        if particle["cell_ID"] == -1:
            event = EVENT_LOST

    # The top cell
    cell = mcdc["cells"][particle["cell_ID"]]

    # Recursively check cells until material cell is found (or the particle is lost)
    while event != EVENT_LOST:
        # Distance to nearest surface
        d_surface, surface_ID = distance_to_nearest_surface(particle, cell, mcdc)

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

            # Apply rotation
            if cell["fill_rotated"]:
                _rotate_particle(particle, cell["rotation"])

            # Universe cell?
            if cell["fill_type"] == FILL_UNIVERSE:
                # Get universe ID
                universe_ID = cell["fill_ID"]

            # Lattice cell?
            elif cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = mcdc["lattices"][cell["fill_ID"]]

                # Distance to lattice grid
                d_lattice = mesh.uniform.get_crossing_distance(particle, speed, lattice)

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
                ix, iy, iz, it, outside = mesh.uniform.get_indices(particle, lattice)
                if outside:
                    event = EVENT_LOST
                    continue
                universe_ID = lattice["universe_IDs"][ix, iy, iz]

                # Lattice-translate the particle
                particle["x"] -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
                particle["y"] -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
                particle["z"] -= lattice["z0"] + (iz + 0.5) * lattice["dz"]

            # Get inner cell
            cell_ID = get_cell(particle, universe_ID, mcdc)
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
        report_lost(particle)

    # Assign particle event
    particle["event"] = event

    return distance


@njit
def locate_particle(particle, mcdc):
    """
    Set particle cell and material IDs
    Return False if particle is lost

    This is similar to inspect_geometry, except that distance to nearest surface
    or/and lattice grid and the respective boundary event are not determined.
    """
    # TODO: add universe cell (besides material and lattice cells)

    # Store particle global coordinate
    # (particle will be temporarily translated and rotated)
    x_global = particle["x"]
    y_global = particle["y"]
    z_global = particle["z"]
    t_global = particle["t"]
    ux_global = particle["ux"]
    uy_global = particle["uy"]
    uz_global = particle["uz"]
    speed = physics.get_speed(particle, mcdc)

    particle_is_lost = False

    # Find top cell from root universe if unknown
    if particle["cell_ID"] == -1:
        particle["cell_ID"] = get_cell(particle, UNIVERSE_ROOT, mcdc)

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

            # Apply rotation
            if cell["fill_rotated"]:
                _rotate_particle(particle, cell["rotation"])

            # Universe cell?
            if cell["fill_type"] == FILL_UNIVERSE:
                # Get universe ID
                universe_ID = cell["fill_ID"]

            # Lattice cell?
            elif cell["fill_type"] == FILL_LATTICE:
                # Get lattice
                lattice = mcdc["lattices"][cell["fill_ID"]]

                # Get universe ID
                ix, iy, iz, it, outside = mesh.uniform.get_indices(particle, lattice)
                if outside:
                    particle_is_lost = True
                    continue
                universe_ID = lattice["universe_IDs"][ix, iy, iz]

                # Lattice-translate the particle
                particle["x"] -= lattice["x0"] + (ix + 0.5) * lattice["dx"]
                particle["y"] -= lattice["y0"] + (iy + 0.5) * lattice["dy"]
                particle["z"] -= lattice["z0"] + (iz + 0.5) * lattice["dz"]

            # Get inner cell
            cell_ID = get_cell(particle, universe_ID, mcdc)
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
        report_lost(particle)

    return not particle_is_lost


@nb.njit
def _rotate_particle(particle, rotation):
    # Particle initial coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Rotation matrix
    xx, xy, xz, yx, yy, yz, zx, zy, zz = _rotation_matrix(rotation)

    # Rotate
    x_rotated = x * xx + y * xy + z * xz
    y_rotated = x * yx + y * yy + z * yz
    z_rotated = x * zx + y * zy + z * zz
    ux_rotated = ux * xx + uy * xy + uz * xz
    uy_rotated = ux * yx + uy * yy + uz * yz
    uz_rotated = ux * zx + uy * zy + uz * zz

    # Assign the rotated coordinate
    particle["x"] = x_rotated
    particle["y"] = y_rotated
    particle["z"] = z_rotated
    particle["ux"] = ux_rotated
    particle["uy"] = uy_rotated
    particle["uz"] = uz_rotated


@nb.njit
def _rotation_matrix(rotation):
    phi = rotation[0]
    theta = rotation[1]
    psi = rotation[2]

    xx = math.cos(theta) * math.cos(psi)
    xy = -math.cos(phi) * math.sin(psi) + math.sin(phi) * math.sin(theta) * math.cos(
        psi
    )
    xz = math.sin(phi) * math.sin(psi) + math.cos(phi) * math.sin(theta) * math.cos(psi)

    yx = math.cos(theta) * math.sin(psi)
    yy = math.cos(phi) * math.cos(psi) + math.sin(phi) * math.sin(theta) * math.sin(psi)
    yz = -math.sin(phi) * math.cos(psi) + math.cos(phi) * math.sin(theta) * math.sin(
        psi
    )

    zx = -math.sin(theta)
    zy = math.sin(phi) * math.cos(theta)
    zz = math.cos(phi) * math.cos(theta)

    return xx, xy, xz, yx, yy, yz, zx, zy, zz


# ======================================================================================
# Particle locator
# ======================================================================================


@njit
def get_cell(particle, universe_ID, mcdc):
    """
    Find and return particle cell ID in the given universe
    Return -1 if particle is lost
    """
    universe = mcdc["universes"][universe_ID]

    # Access universe cell data
    idx = universe["cell_data_idx"]
    N_cell = universe["N_cell"]

    # Check over all cells in the universe
    idx_end = idx + N_cell
    while idx < idx_end:
        cell_ID = mcdc["universes_data_cell"][idx]
        cell = mcdc["cells"][cell_ID]
        if check_cell(particle, cell, mcdc):
            return cell["ID"]
        idx += 1

    # Particle is not found
    return -1


@njit
def check_cell(particle, cell, mcdc):
    """
    Check if the particle is inside the cell
    """
    # Access Region RPN data
    idx = cell["region_data_idx"]
    N_token = cell["N_region"]

    # Create local value array
    value_struct = local.RPN_array()
    value = value_struct["values"]
    N_value = 0

    # March forward through RPN tokens
    idx_end = idx + N_token
    while idx < idx_end:
        token = mcdc["cells_data_region"][idx]

        if token >= 0:
            surface = mcdc["surfaces"][token]
            value[N_value] = check_surface_sense(particle, surface, mcdc)
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
def check_surface_sense(particle, surface, mcdc):
    """
    Check on which side of the surface the particle is
        - Return True if positive side
        - Return False otherwise
    Particle direction is used if coincide within the tolerance
    """
    result = surface_evaluate(particle, surface)

    # Check if coincident on the surface
    if abs(result) < COINCIDENCE_TOLERANCE:
        # Determine sense based on the direction
        return surface_normal_component(particle, surface, mcdc) > 0.0

    return result > 0.0


@njit
def report_lost(particle):
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
def distance_to_nearest_surface(particle, cell, mcdc):
    """
    Determine the nearest cell surface and the distance to it
    """
    distance = INF
    surface_ID = -1

    # Access cell surface data
    idx = cell["surface_data_idx"]
    N_surface = cell["N_surface"]

    # Iterate over all surfaces
    idx_end = idx + N_surface
    while idx < idx_end:
        candidate_surface_ID = mcdc["cells_data_surface"][idx]
        surface = mcdc["surfaces"][candidate_surface_ID]
        d = surface_distance(particle, surface, mcdc)
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
def surface_evaluate(particle, surface):
    """
    Evaluate the surface equation wrt the particle coordinate
    """

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
        if not surface["moving"]:
            return result
        else:
            # Get move index
            N_move = surface["N_move"]
            time_grid = surface["move_time_grid"]
            idx = binary_search_with_length(t, time_grid, N_move)
            # Coinciding cases
            if abs(time_grid[idx + 1] - t) < COINCIDENCE_TOLERANCE:
                idx += 1

            # Get translation points
            translation_start = surface["move_translations"][idx]
            translation_end = surface["move_translations"][idx + 1]

            # Interpolate translation
            dx = translation_end[0] - translation_start[0]
            dy = translation_end[1] - translation_start[1]
            dz = translation_end[2] - translation_start[2]
            dt = time_grid[idx + 1] - time_grid[idx]
            tt = t - time_grid[idx]

            tr_x = translation_start[0] + dx / dt * tt
            tr_y = translation_start[1] + dy / dt * tt
            tr_z = translation_start[2] + dz / dt * tt

            return result - (G * tr_x + H * tr_y + I * tr_z)

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
def surface_distance(particle, surface, mcdc):
    """
    Return particle distance to surface
    """
    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    t = particle["t"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]
    speed = physics.get_speed(particle, mcdc)

    # Check if coincident and leaving the surface forever
    evaluation = surface_evaluate(particle, surface)
    coincident = abs(evaluation) < COINCIDENCE_TOLERANCE
    if coincident:
        if surface["linear"]:
            if not surface["moving"]:
                return INF
        elif surface_normal_component(particle, surface, mcdc) > 0.0:
            return INF

    # Surface coefficients
    G = surface["G"]
    H = surface["H"]
    I = surface["I"]
    J = surface["J"]

    # Distance to linear surface
    if surface["linear"]:
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
                distance = -surface_evaluate(particle, surface) / relative_velocity
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


@njit
def surface_bc(particle, surface):
    """
    Apply surface boundary condition to the particle
    """
    if surface["BC"] == BC_VACUUM:
        particle["alive"] = False
    elif surface["BC"] == BC_REFLECTIVE:
        surface_reflect(particle, surface)


@njit
def surface_reflect(particle, surface):
    """
    Surface-reflect the particle
    """
    # Particle coordinate
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    nx, ny, nz = surface_normal(particle, surface)

    c = 2.0 * (nx * ux + ny * uy + nz * uz)

    particle["ux"] = ux - c * nx
    particle["uy"] = uy - c * ny
    particle["uz"] = uz - c * nz


@njit
def surface_normal(particle, surface):
    """
    Get the surface outward-normal vector at the particle coordinate
    """
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
def surface_normal_component(particle, surface, mcdc):
    """
    Get the surface outward-normal component of the particle
    (dot product of the two directional vectors)
    """
    # Surface outward-normal vector
    nx, ny, nz = surface_normal(particle, surface)

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
        speed = physics.get_speed(particle, mcdc)
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


# ======================================================================================
# Miscellanies
# ======================================================================================


@njit
def check_coincidence(value_1, value_2):
    """
    Check if two values are within coincidence tolerance
    """
    return abs(value_1 - value_2) < COINCIDENCE_TOLERANCE
