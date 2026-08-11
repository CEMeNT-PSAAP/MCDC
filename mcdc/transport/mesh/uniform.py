import math

from numba import njit

from mcdc.constant import COINCIDENCE_TOLERANCE, INF


@njit
def get_indices(particle_container, uniform_mesh):
    """
    Get uniform_mesh indices given the particle coordinate
    """
    particle = particle_container[0]

    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Mesh parameters
    x0 = uniform_mesh["x0"]
    y0 = uniform_mesh["y0"]
    z0 = uniform_mesh["z0"]
    dx = uniform_mesh["dx"]
    dy = uniform_mesh["dy"]
    dz = uniform_mesh["dz"]
    Nx = uniform_mesh["Nx"]
    Ny = uniform_mesh["Ny"]
    Nz = uniform_mesh["Nz"]
    x_last = x0 + Nx * dx
    y_last = y0 + Ny * dy
    z_last = z0 + Nz * dz

    # x-axis
    if (
        # Outside the mesh
        x < x0 - COINCIDENCE_TOLERANCE
        or x > x_last + COINCIDENCE_TOLERANCE
        # At the outermost-grid but moving away
        or (abs(x - x0) < COINCIDENCE_TOLERANCE and ux < 0.0)
        or (abs(x - x_last) < COINCIDENCE_TOLERANCE and ux > 0.0)
    ):
        ix = -1
    else:
        ix = _grid_index(x, ux, x0, dx, COINCIDENCE_TOLERANCE)

    # y-axis
    if (
        # Outside the mesh
        y < y0 - COINCIDENCE_TOLERANCE
        or y > y_last + COINCIDENCE_TOLERANCE
        # At the outermost-grid but moving away
        or (abs(y - y0) < COINCIDENCE_TOLERANCE and uy < 0.0)
        or (abs(y - y_last) < COINCIDENCE_TOLERANCE and uy > 0.0)
    ):
        iy = -1
    else:
        iy = _grid_index(y, uy, y0, dy, COINCIDENCE_TOLERANCE)

    # z-axis
    if (
        # Outside the mesh
        z < z0 - COINCIDENCE_TOLERANCE
        or z > z_last + COINCIDENCE_TOLERANCE
        # At the outermost-grid but moving away
        or (abs(z - z0) < COINCIDENCE_TOLERANCE and uz < 0.0)
        or (abs(z - z_last) < COINCIDENCE_TOLERANCE and uz > 0.0)
    ):
        iz = -1
    else:
        iz = _grid_index(z, uz, z0, dz, COINCIDENCE_TOLERANCE)

    return ix, iy, iz


@njit
def get_crossing_distance(particle_container, speed, uniform_mesh):
    """
    Get distance for the particle, moving with the given speed,
    to cross the nearest grid of the uniform_mesh
    """
    particle = particle_container[0]

    # Particle coordinate
    x = particle["x"]
    y = particle["y"]
    z = particle["z"]
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Mesh parameters
    x0 = uniform_mesh["x0"]
    y0 = uniform_mesh["y0"]
    z0 = uniform_mesh["z0"]
    dx = uniform_mesh["dx"]
    dy = uniform_mesh["dy"]
    dz = uniform_mesh["dz"]
    Nx = uniform_mesh["Nx"]
    Ny = uniform_mesh["Ny"]
    Nz = uniform_mesh["Nz"]
    x_last = x0 + Nx * dx
    y_last = y0 + Ny * dy
    z_last = z0 + Nz * dz

    # Check if particle is outside the mesh grid and moving away
    if (
        (x < x0 + COINCIDENCE_TOLERANCE and ux < 0.0)
        or (x > x_last - COINCIDENCE_TOLERANCE and ux > 0.0)
        or (y < y0 + COINCIDENCE_TOLERANCE and uy < 0.0)
        or (y > y_last - COINCIDENCE_TOLERANCE and uy > 0.0)
        or (z < z0 + COINCIDENCE_TOLERANCE and uz < 0.0)
        or (z > z_last - COINCIDENCE_TOLERANCE and uz > 0.0)
    ):
        return INF

    d = INF
    d = min(d, _grid_distance(x, ux, x0, dx, COINCIDENCE_TOLERANCE))
    d = min(d, _grid_distance(y, uy, y0, dy, COINCIDENCE_TOLERANCE))
    d = min(d, _grid_distance(z, uz, z0, dz, COINCIDENCE_TOLERANCE))
    return d


@njit
def _grid_index(value, direction, start, width, tolerance):
    """
    Get grid index given the value and the direction

    Direction is used to tiebreak when the value is at a grid point
    (within tolerance).
    Note: It assumes the value is inside the grid.
    """
    idx = int(math.floor((value + tolerance - start) / width))

    # Coinciding cases
    if abs(start + width * idx - value) < tolerance:
        if direction < 0.0:
            idx -= 1

    return idx


@njit
def _grid_distance(value, direction, start, width, tolerance):
    """
    Get distance to nearest grid given a value and direction

    Direction is used to tiebreak when the value is at a grid point
    (within tolerance).
    Note: It assumes that a grid must be hit
    """
    if direction == 0.0:
        return INF

    idx = int(math.floor((value + tolerance - start) / width))

    # Coinciding cases
    if abs(start + width * idx - value) < tolerance:
        if direction < 0.0:
            idx -= 1

    if direction > 0.0:
        idx += 1

    dist = (start + width * idx - value) / direction

    return dist
