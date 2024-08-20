import numpy as np

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)

from mcdc.mesh import (
    get_indices,
    get_grid_index,
    nearest_distance_to_grid,
)


def test_get_grid_index():
    grid = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    tiny = COINCIDENCE_TOLERANCE * 0.8

    # Inside bin, going right
    assert np.isclose(get_grid_index(-3.2, 0.4, grid), 0)
    assert np.isclose(get_grid_index(3.2, 0.4, grid), 5)

    # Inside bin, going left
    assert np.isclose(get_grid_index(-3.2, -0.4, grid), 0)
    assert np.isclose(get_grid_index(3.2, -0.4, grid), 5)

    # At internal grid, going right
    assert np.isclose(get_grid_index(3.0, 0.4, grid), 5)
    assert np.isclose(get_grid_index(-3.0, 0.4, grid), 1)

    # At internal grid, going left
    assert np.isclose(get_grid_index(3.0, -0.4, grid), 4)
    assert np.isclose(get_grid_index(-3.0, -0.4, grid), 0)

    # At left-most grid, going right
    assert np.isclose(get_grid_index(-6.0, 0.4, grid), 0)

    # At right-most grid, going left
    assert np.isclose(get_grid_index(6.0, -0.4, grid), 5)

    # At internal grid (within tolerance), going right
    assert np.isclose(get_grid_index(3.0 + tiny, 0.4, grid), 5)
    assert np.isclose(get_grid_index(3.0 - tiny, 0.4, grid), 5)
    assert np.isclose(get_grid_index(-3.0 + tiny, 0.4, grid), 1)
    assert np.isclose(get_grid_index(-3.0 - tiny, 0.4, grid), 1)

    # At internal grid (within tolerance), going left
    assert np.isclose(get_grid_index(3.0 + tiny, -0.4, grid), 4)
    assert np.isclose(get_grid_index(3.0 - tiny, -0.4, grid), 4)
    assert np.isclose(get_grid_index(-3.0 + tiny, -0.4, grid), 0)
    assert np.isclose(get_grid_index(-3.0 - tiny, -0.4, grid), 0)

    # At left-most grid (within tolerance), going right
    assert np.isclose(get_grid_index(-6.0 + tiny, 0.4, grid), 0)
    assert np.isclose(get_grid_index(-6.0 - tiny, 0.4, grid), 0)

    # At right-most grid (within tolerance), going left
    assert np.isclose(get_grid_index(6.0 + tiny, -0.4, grid), 5)
    assert np.isclose(get_grid_index(6.0 - tiny, -0.4, grid), 5)


def test_get_indices():
    mesh = {}
    mesh["x"] = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    mesh["y"] = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    mesh["z"] = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    mesh["t"] = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    tiny = COINCIDENCE_TOLERANCE * 0.8

    def particle(x, y, z, t, ux, uy, uz):
        return {"x": x, "y": y, "z": z, "t": t, "ux": ux, "uy": uy, "uz": uz}

    # Inside bin
    ix, iy, iz, it, outside = get_indices(
        particle(-5.0, -2.0, -0.5, 4.0, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 0 and iy == 1 and iz == 2 and it == 5 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-5.0, -2.0, -0.5, 4.0, -1.0, -1.0, -1.0), mesh
    )
    assert ix == 0 and iy == 1 and iz == 2 and it == 5 and not outside

    # Outside
    ix, iy, iz, it, outside = get_indices(
        particle(-5.0, -2.0, -0.5, 10.0, 1.0, 1.0, 1.0), mesh
    )
    assert ix == -1 and iy == -1 and iz == -1 and it == -1 and outside
    ix, iy, iz, it, outside = get_indices(
        particle(-5.0, -2.0, -0.5, 10.0, -1.0, -1.0, -1.0), mesh
    )
    assert ix == -1 and iy == -1 and iz == -1 and it == -1 and outside

    # At internal grid
    ix, iy, iz, it, outside = get_indices(
        particle(-3.0, -1.0, 1.0, 3.0, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 1 and iy == 2 and iz == 4 and it == 5 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-3.0, -1.0, 1.0, 3.0, -1.0, -1.0, -1.0), mesh
    )
    assert ix == 0 and iy == 1 and iz == 3 and it == 5 and not outside

    # At left-most grid
    ix, iy, iz, it, outside = get_indices(
        particle(-6.0, -6.0, -6.0, 0.0, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 0 and iy == 0 and iz == 0 and it == 3 and not outside

    # At right-most grid
    ix, iy, iz, it, outside = get_indices(
        particle(6.0, 6.0, 6.0, 0.0, -1.0, -1.0, -1.0), mesh
    )
    assert ix == 5 and iy == 5 and iz == 5 and it == 3 and not outside

    # At internal grid (within tolerance)
    ix, iy, iz, it, outside = get_indices(
        particle(-3.0 + tiny, -1.0 + tiny, 1.0 + tiny, 3.0 + tiny, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 1 and iy == 2 and iz == 4 and it == 5 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-3.0 - tiny, -1.0 - tiny, 1.0 - tiny, 3.0 - tiny, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 1 and iy == 2 and iz == 4 and it == 5 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-3.0 + tiny, -1.0 + tiny, 1.0 + tiny, 3.0 + tiny, -1.0, -1.0, -1.0),
        mesh,
    )
    assert ix == 0 and iy == 1 and iz == 3 and it == 5 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-3.0 - tiny, -1.0 - tiny, 1.0 - tiny, 3.0 - tiny, -1.0, -1.0, -1.0),
        mesh,
    )
    assert ix == 0 and iy == 1 and iz == 3 and it == 5 and not outside

    # At left-most grid (within tolerance)
    ix, iy, iz, it, outside = get_indices(
        particle(-6.0 + tiny, -6.0 + tiny, -6.0 + tiny, 0.0 + tiny, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 0 and iy == 0 and iz == 0 and it == 3 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-6.0 - tiny, -6.0 - tiny, -6.0 - tiny, 0.0 - tiny, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 0 and iy == 0 and iz == 0 and it == 3 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-6.0 + tiny, -6.0 + tiny, -6.0 + tiny, 0.0 + tiny, -1.0, -1.0, -1.0),
        mesh,
    )
    assert ix == -1 and iy == -1 and iz == -1 and it == -1 and outside
    ix, iy, iz, it, outside = get_indices(
        particle(-6.0 - tiny, -6.0 - tiny, -6.0 - tiny, 0.0 - tiny, -1.0, -1.0, -1.0),
        mesh,
    )
    assert ix == -1 and iy == -1 and iz == -1 and it == -1 and outside

    # At right-most grid (within tolerance)
    ix, iy, iz, it, outside = get_indices(
        particle(6.0 + tiny, 6.0 + tiny, 6.0 + tiny, 0.0 + tiny, 1.0, 1.0, 1.0), mesh
    )
    assert ix == -1 and iy == -1 and iz == -1 and it == -1 and outside
    ix, iy, iz, it, outside = get_indices(
        particle(6.0 - tiny, 6.0 - tiny, 6.0 - tiny, 0.0 - tiny, 1.0, 1.0, 1.0), mesh
    )
    assert ix == -1 and iy == -1 and iz == -1 and it == -1 and outside
    ix, iy, iz, it, outside = get_indices(
        particle(6.0 + tiny, 6.0 + tiny, 6.0 + tiny, 0.0 + tiny, -1.0, -1.0, -1.0), mesh
    )
    assert ix == 5 and iy == 5 and iz == 5 and it == 3 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(6.0 - tiny, 6.0 - tiny, 6.0 - tiny, 0.0 - tiny, -1.0, -1.0, -1.0), mesh
    )
    assert ix == 5 and iy == 5 and iz == 5 and it == 3 and not outside


def test_nearest_distance_to_grid():
    grid = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])
    tiny = COINCIDENCE_TOLERANCE * 0.8

    # Inside bin, going right
    assert np.isclose(nearest_distance_to_grid(-3.2, 0.4, grid), 0.2 / 0.4)
    assert np.isclose(nearest_distance_to_grid(3.2, 0.4, grid), 2.8 / 0.4)

    # Inside bin, going left
    assert np.isclose(nearest_distance_to_grid(-3.2, -0.4, grid), 2.8 / 0.4)
    assert np.isclose(nearest_distance_to_grid(3.2, -0.4, grid), 0.2 / 0.4)

    # Outside, moving away
    assert np.isclose(nearest_distance_to_grid(8.0, 0.4, grid), INF)
    assert np.isclose(nearest_distance_to_grid(-8.0, -0.4, grid), INF)

    # Outside, moving closer
    assert np.isclose(nearest_distance_to_grid(8.0, -0.4, grid), 2.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(-8.0, 0.4, grid), 2.0 / 0.4)

    # At internal grid, going right
    assert np.isclose(nearest_distance_to_grid(3.0, 0.4, grid), 3.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(-3.0, 0.4, grid), 2.0 / 0.4)

    # At internal grid, going left
    assert np.isclose(nearest_distance_to_grid(3.0, -0.4, grid), 2.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(-3.0, -0.4, grid), 3.0 / 0.4)

    # At left-most grid, going right
    assert np.isclose(nearest_distance_to_grid(-6.0, 0.4, grid), 3.0 / 0.4)

    # At left-most grid, going left
    assert np.isclose(nearest_distance_to_grid(-6.0, -0.4, grid), INF)

    # At right-most grid, going right
    assert np.isclose(nearest_distance_to_grid(6.0, 0.4, grid), INF)

    # At right-most grid, going left
    assert np.isclose(nearest_distance_to_grid(6.0, -0.4, grid), 3.0 / 0.4)

    # At internal grid (within tolerance), going right
    assert np.isclose(nearest_distance_to_grid(3.0 + tiny, 0.4, grid), 3.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(3.0 - tiny, 0.4, grid), 3.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(-3.0 + tiny, 0.4, grid), 2.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(-3.0 - tiny, 0.4, grid), 2.0 / 0.4)

    # At internal grid (within tolerance), going left
    assert np.isclose(nearest_distance_to_grid(3.0 + tiny, -0.4, grid), 2.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(3.0 - tiny, -0.4, grid), 2.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(-3.0 + tiny, -0.4, grid), 3.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(-3.0 - tiny, -0.4, grid), 3.0 / 0.4)

    # At left-most grid (within tolerance), going right
    assert np.isclose(nearest_distance_to_grid(-6.0 + tiny, 0.4, grid), 3.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(-6.0 - tiny, 0.4, grid), 3.0 / 0.4)

    # At left-most grid (within tolerance), going left
    assert np.isclose(nearest_distance_to_grid(-6.0 + tiny, -0.4, grid), INF)
    assert np.isclose(nearest_distance_to_grid(-6.0 - tiny, -0.4, grid), INF)

    # At right-most grid (within tolerance), going right
    assert np.isclose(nearest_distance_to_grid(6.0 + tiny, 0.4, grid), INF)
    assert np.isclose(nearest_distance_to_grid(6.0 - tiny, 0.4, grid), INF)

    # At right-most grid (within tolerance), going left
    assert np.isclose(nearest_distance_to_grid(6.0 + tiny, -0.4, grid), 3.0 / 0.4)
    assert np.isclose(nearest_distance_to_grid(6.0 - tiny, -0.4, grid), 3.0 / 0.4)
