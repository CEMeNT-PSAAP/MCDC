import math

from numba import int64, njit

from mcdc.constant import COINCIDENCE_TOLERANCE, COINCIDENCE_TOLERANCE_TIME, INF


@njit
def get_indices(particle_container, mesh):
    """
    Get mesh indices given the particle coordinate
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

    # Mesh parameters
    x0 = mesh["x0"]
    y0 = mesh["y0"]
    z0 = mesh["z0"]
    t0 = mesh["t0"]
    dx = mesh["dx"]
    dy = mesh["dy"]
    dz = mesh["dz"]
    dt = mesh["dt"]
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    Nt = mesh["Nt"]
    x_last = x0 + Nx * dx
    y_last = y0 + Ny * dy
    z_last = z0 + Nz * dz
    t_last = t0 + Nt * dt

    # Check if particle is outside the mesh grid
    outside = False
    if (
        # Outside the mesh condition
        x < x0 - COINCIDENCE_TOLERANCE
        or x > x_last + COINCIDENCE_TOLERANCE
        or y < y0 - COINCIDENCE_TOLERANCE
        or y > y_last + COINCIDENCE_TOLERANCE
        or z < z0 - COINCIDENCE_TOLERANCE
        or z > z_last + COINCIDENCE_TOLERANCE
        or t < t0 - COINCIDENCE_TOLERANCE_TIME
        or t > t_last + COINCIDENCE_TOLERANCE_TIME
        # At the outermost-grid but moving away
        or (abs(x - x0) < COINCIDENCE_TOLERANCE and ux < 0.0)
        or (abs(x - x_last) < COINCIDENCE_TOLERANCE and ux > 0.0)
        or (abs(y - y0) < COINCIDENCE_TOLERANCE and uy < 0.0)
        or (abs(y - y_last) < COINCIDENCE_TOLERANCE and uy > 0.0)
        or (abs(z - z0) < COINCIDENCE_TOLERANCE and uz < 0.0)
        or (abs(z - z_last) < COINCIDENCE_TOLERANCE and uz > 0.0)
        or (abs(t - t_last) < COINCIDENCE_TOLERANCE_TIME)
    ):
        outside = True
        return -1, -1, -1, -1, outside

    ix = _grid_index(x, ux, x0, dx, COINCIDENCE_TOLERANCE)
    iy = _grid_index(y, uy, y0, dy, COINCIDENCE_TOLERANCE)
    iz = _grid_index(z, uz, z0, dz, COINCIDENCE_TOLERANCE)
    it = _grid_index(
        t, 1.0, t0, dt, COINCIDENCE_TOLERANCE_TIME
    )  # Particle always moves forward in time

    return ix, iy, iz, it, outside


@njit
def get_crossing_distance(particle_container, speed, mesh):
    """
    Get distance for the particle, moving with the given speed,
    to cross the nearest grid of the mesh
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

    # Mesh parameters
    x0 = mesh["x0"]
    y0 = mesh["y0"]
    z0 = mesh["z0"]
    t0 = mesh["t0"]
    dx = mesh["dx"]
    dy = mesh["dy"]
    dz = mesh["dz"]
    dt = mesh["dt"]
    Nx = mesh["Nx"]
    Ny = mesh["Ny"]
    Nz = mesh["Nz"]
    Nt = mesh["Nt"]
    x_last = x0 + Nx * dx
    y_last = y0 + Ny * dy
    z_last = z0 + Nz * dz
    t_last = t0 + Nt * dt

    # Check if particle is outside the mesh grid and moving away
    outside = False
    if (
        (t > t_last - COINCIDENCE_TOLERANCE_TIME)
        or (x < x0 + COINCIDENCE_TOLERANCE and ux < 0.0)
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
    d = min(d, _grid_distance(t, 1.0 / speed, t0, dt, COINCIDENCE_TOLERANCE_TIME))
    return d


@njit
def _grid_index(value, direction, start, width, tolerance):
    """
    Get grid index given the value and the direction

    Direction is used to tiebreak when the value is at a grid point
    (within tolerance).
    Note: It assumes the value is inside the grid.
    """
    idx = int64(math.floor((value + tolerance - start) / width))

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

    idx = int64(math.floor((value + tolerance - start) / width))

    # Coinciding cases
    if abs(start + width * idx - value) < tolerance:
        if direction < 0.0:
            idx -= 1

    if direction > 0.0:
        idx += 1

    dist = (start + width * idx - value) / direction

    return dist
