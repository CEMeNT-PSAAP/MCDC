from numba import njit

from mcdc.algorithm import binary_search
from mcdc.constant import COINCIDENCE_TOLERANCE, INF


@njit
def get_indices(particle, mesh):
    """
    Get mesh indices given the coordinate

    Direction is used to tiebreak when the value is at a grid point
    (within tolerance).
    """
    # Check if coordinate is outside the mesh grid
    outside = False
    if (
        particle["x"] < mesh["x"][0] - COINCIDENCE_TOLERANCE
        or particle["x"] > mesh["x"][-1] + COINCIDENCE_TOLERANCE
        or particle["y"] < mesh["y"][0] - COINCIDENCE_TOLERANCE
        or particle["y"] > mesh["y"][-1] + COINCIDENCE_TOLERANCE
        or particle["z"] < mesh["z"][0] - COINCIDENCE_TOLERANCE
        or particle["z"] > mesh["z"][-1] + COINCIDENCE_TOLERANCE
        or particle["t"] < mesh["t"][0] - COINCIDENCE_TOLERANCE
        or particle["t"] > mesh["t"][-1] + COINCIDENCE_TOLERANCE
        or (
            abs(particle["x"] - mesh["x"][0]) < COINCIDENCE_TOLERANCE
            and particle["ux"] < 0.0
        )
        or (
            abs(particle["x"] - mesh["x"][-1]) < COINCIDENCE_TOLERANCE
            and particle["ux"] > 0.0
        )
        or (
            abs(particle["y"] - mesh["y"][0]) < COINCIDENCE_TOLERANCE
            and particle["uy"] < 0.0
        )
        or (
            abs(particle["y"] - mesh["y"][-1]) < COINCIDENCE_TOLERANCE
            and particle["uy"] > 0.0
        )
        or (
            abs(particle["z"] - mesh["z"][0]) < COINCIDENCE_TOLERANCE
            and particle["uz"] < 0.0
        )
        or (
            abs(particle["z"] - mesh["z"][-1]) < COINCIDENCE_TOLERANCE
            and particle["uz"] > 0.0
        )
        or (abs(particle["t"] - mesh["t"][-1]) < COINCIDENCE_TOLERANCE and 1.0 > 0.0)
    ):
        outside = True
        return -1, -1, -1, -1, outside

    ix = get_grid_index(particle["x"], particle["ux"], mesh["x"])
    iy = get_grid_index(particle["y"], particle["uy"], mesh["y"])
    iz = get_grid_index(particle["z"], particle["uz"], mesh["z"])
    it = get_grid_index(particle["t"], 1.0, mesh["t"])

    return ix, iy, iz, it, outside


@njit
def get_grid_index(value, direction, grid):
    """
    Get grid index given a value and direction

    Direction is used to tiebreak when the value is at a grid point
    (within tolerance).
    It assumes value is inside the grid.
    """
    idx = binary_search(value, grid)

    # Coinciding cases
    if direction > 0.0:
        if abs(grid[idx + 1] - value) < COINCIDENCE_TOLERANCE:
            idx += 1
    else:
        if abs(grid[idx] - value) < COINCIDENCE_TOLERANCE:
            idx -= 1
    return idx


@njit
def nearest_distance_to_grid(value, direction, grid):
    """
    Get the nearest distance to grid given a value and direction

    Direction is used to tiebreak when the value is at a grid point
    (within tolerance).
    """
    if direction == 0.0:
        return INF

    idx = binary_search(value, grid)

    if direction > 0.0:
        idx += 1

    # Edge cases
    if idx == -1:
        idx += 1
    if idx == len(grid):
        idx -= 1

    # Coinciding cases
    if abs(grid[idx] - value) < COINCIDENCE_TOLERANCE:
        if direction > 0.0:
            idx += 1

            if idx == len(grid):
                # Right-most, going right
                return INF
        else:
            if value == grid[0]:
                # Left-most, going left
                return INF
            else:
                idx -= 1

    dist = (grid[idx] - value) / direction

    # Moving away from mesh?
    if dist < 0.0:
        dist = INF
    return dist
