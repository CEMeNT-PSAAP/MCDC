from numba import njit


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
