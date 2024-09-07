import numpy as np

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)

from mcdc.mesh.uniform import (
    get_indices,
    get_crossing_distance,
    _grid_index,
    _grid_distance,
)


def particle(x, y, z, t, ux, uy, uz):
    return [{"x": x, "y": y, "z": z, "t": t, "ux": ux, "uy": uy, "uz": uz}]


grid = np.array([-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0])

mesh = {}
mesh["x0"] = -6.0
mesh["y0"] = -6.0
mesh["z0"] = -6.0
mesh["t0"] = -6.0
mesh["dx"] = 2.0
mesh["dy"] = 2.0
mesh["dz"] = 2.0
mesh["dt"] = 2.0
mesh["Nx"] = 6
mesh["Ny"] = 6
mesh["Nz"] = 6
mesh["Nt"] = 6

tiny = COINCIDENCE_TOLERANCE * 0.8


def test_get_indices():
    # Inside bin
    ix, iy, iz, it, outside = get_indices(
        particle(-5.2, -3.2, -1.2, 1.2, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 0 and iy == 1 and iz == 2 and it == 3 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-5.2, -3.2, -1.2, 1.2, -1.0, -1.0, -1.0), mesh
    )
    assert ix == 0 and iy == 1 and iz == 2 and it == 3 and not outside

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
        particle(-4.0, -2.0, 0.0, 2.0, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 1 and iy == 2 and iz == 3 and it == 4 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-4.0, -2.0, 0.0, 2.0, -1.0, -1.0, -1.0), mesh
    )
    assert ix == 0 and iy == 1 and iz == 2 and it == 4 and not outside

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
        particle(-4.0 + tiny, -2.0 + tiny, 0.0 + tiny, 2.0 + tiny, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 1 and iy == 2 and iz == 3 and it == 4 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-4.0 - tiny, -2.0 - tiny, 0.0 - tiny, 2.0 - tiny, 1.0, 1.0, 1.0), mesh
    )
    assert ix == 1 and iy == 2 and iz == 3 and it == 4 and not outside

    ix, iy, iz, it, outside = get_indices(
        particle(-4.0 + tiny, -2.0 + tiny, 0.0 + tiny, 2.0 + tiny, -1.0, -1.0, -1.0),
        mesh,
    )
    assert ix == 0 and iy == 1 and iz == 2 and it == 4 and not outside
    ix, iy, iz, it, outside = get_indices(
        particle(-4.0 - tiny, -2.0 - tiny, 0.0 - tiny, 2.0 - tiny, -1.0, -1.0, -1.0),
        mesh,
    )
    assert ix == 0 and iy == 1 and iz == 2 and it == 4 and not outside

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


-6, -4, -2, 0, 2, 4, 6


def test_get_corssing_distance():
    # Inside bin
    distance = get_crossing_distance(
        particle(-4.2, -3.2, -1.2, 1.2, 0.4, 0.4, 0.4), 1.0 / 0.4, mesh
    )
    assert np.isclose(distance, 0.2 / 0.4)
    distance = get_crossing_distance(
        particle(-4.2, -3.2, 0.1, 1.2, 0.4, 0.4, -0.4), 1.0 / 0.4, mesh
    )
    assert np.isclose(distance, 0.1 / 0.4)

    # Outside, moving away
    distance = get_crossing_distance(
        particle(8.2, -3.3, -3.4, -3.5, 0.4, 0.4, 0.4), 1.0 / 0.4, mesh
    )
    assert np.isclose(distance, INF)

    # Outside, moving closer
    distance = get_crossing_distance(
        particle(-6.1, -3.3, -3.4, -3.5, 0.4, 0.4, 0.4), 1.0 / 0.4, mesh
    )
    assert np.isclose(distance, 0.1 / 0.4)

    # At internal grid
    distance = get_crossing_distance(
        particle(-4.0, -4.0, -4.0, -4.0, 0.4, 0.3, 0.2), 1.0, mesh
    )
    assert np.isclose(distance, 2.0)
    distance = get_crossing_distance(
        particle(-4.0, -4.0, -4.0, -4.0, -0.4, -0.3, -0.2), -1.0, mesh
    )
    assert np.isclose(distance, 2.0)

    # At left-most grid, going right
    distance = get_crossing_distance(
        particle(-6.0, -3.0, -3.0, -3.0, 1.0, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)

    # At left-most grid, going left
    distance = get_crossing_distance(
        particle(-6.0, -3.0, -3.0, -3.0, -0.1, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, INF)

    # At right-most grid, going right
    distance = get_crossing_distance(
        particle(6.0, -3.0, -3.0, -3.0, 1.0, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, INF)

    # At right-most grid, going left
    distance = get_crossing_distance(
        particle(6.0, -3.0, -3.0, -3.0, -1.0, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)

    # At internal grid (within tolerance)
    distance = get_crossing_distance(
        particle(-4.0 + tiny, -4.0, -4.0, -4.0, 1.0, 0.3, 0.2), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)
    distance = get_crossing_distance(
        particle(-4.0 - tiny, -4.0, -4.0, -4.0, 1.0, 0.3, 0.2), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)
    distance = get_crossing_distance(
        particle(-4.0 + tiny, -3.0, -3.0, -3.0, -1.0, -0.3, -0.2), -1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)
    distance = get_crossing_distance(
        particle(-4.0 - tiny, -3.0, -3.0, -3.0, -1.0, -0.3, -0.2), -1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)

    # At left-most grid, going right (within tolerance)
    distance = get_crossing_distance(
        particle(-6.0 + tiny, -3.0, -3.0, -3.0, 1.0, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)
    distance = get_crossing_distance(
        particle(-6.0 - tiny, -3.0, -3.0, -3.0, 1.0, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)

    # At left-most grid, going left (within tolerance)
    distance = get_crossing_distance(
        particle(-6.0 + tiny, -3.0, -3.0, -3.0, -0.1, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, INF)
    distance = get_crossing_distance(
        particle(-6.0 - tiny, -3.0, -3.0, -3.0, -0.1, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, INF)

    # At right-most grid, going right (within tolerance)
    distance = get_crossing_distance(
        particle(6.0 + tiny, -3.0, -3.0, -3.0, 1.0, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, INF)
    distance = get_crossing_distance(
        particle(6.0 - tiny, -3.0, -3.0, -3.0, 1.0, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, INF)

    # At right-most grid, going left (within tolerance)
    distance = get_crossing_distance(
        particle(6.0 + tiny, -3.0, -3.0, -3.0, -1.0, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)
    distance = get_crossing_distance(
        particle(6.0 - tiny, -3.0, -3.0, -3.0, -1.0, 0.1, 0.1), 1.0 / 0.1, mesh
    )
    assert np.isclose(distance, 2.0)


def test__grid_index():
    # Inside bin, going right
    assert np.isclose(_grid_index(-3.2, 0.4, mesh["x0"], mesh["dx"]), 1)
    assert np.isclose(_grid_index(3.2, 0.4, mesh["x0"], mesh["dx"]), 4)

    # Inside bin, going left
    assert np.isclose(_grid_index(-3.2, -0.4, mesh["x0"], mesh["dx"]), 1)
    assert np.isclose(_grid_index(3.2, -0.4, mesh["x0"], mesh["dx"]), 4)

    # At internal grid, going right
    assert np.isclose(_grid_index(2.0, 0.4, mesh["x0"], mesh["dx"]), 4)
    assert np.isclose(_grid_index(-2.0, 0.4, mesh["x0"], mesh["dx"]), 2)

    # At internal grid, going left
    assert np.isclose(_grid_index(2.0, -0.4, mesh["x0"], mesh["dx"]), 3)
    assert np.isclose(_grid_index(-2.0, -0.4, mesh["x0"], mesh["dx"]), 1)

    # At left-most grid, going right
    assert np.isclose(_grid_index(-6.0, 0.4, mesh["x0"], mesh["dx"]), 0)

    # At right-most grid, going left
    assert np.isclose(_grid_index(6.0, -0.4, mesh["x0"], mesh["dx"]), 5)

    # At internal grid (within tolerance), going right
    assert np.isclose(_grid_index(2.0 + tiny, 0.4, mesh["x0"], mesh["dx"]), 4)
    assert np.isclose(_grid_index(2.0 - tiny, 0.4, mesh["x0"], mesh["dx"]), 4)
    assert np.isclose(_grid_index(-2.0 + tiny, 0.4, mesh["x0"], mesh["dx"]), 2)
    assert np.isclose(_grid_index(-2.0 - tiny, 0.4, mesh["x0"], mesh["dx"]), 2)

    # At internal grid (within tolerance), going left
    assert np.isclose(_grid_index(2.0 + tiny, -0.4, mesh["x0"], mesh["dx"]), 3)
    assert np.isclose(_grid_index(2.0 - tiny, -0.4, mesh["x0"], mesh["dx"]), 3)
    assert np.isclose(_grid_index(-2.0 + tiny, -0.4, mesh["x0"], mesh["dx"]), 1)
    assert np.isclose(_grid_index(-2.0 - tiny, -0.4, mesh["x0"], mesh["dx"]), 1)

    # At left-most grid (within tolerance), going right
    assert np.isclose(_grid_index(-6.0 + tiny, 0.4, mesh["x0"], mesh["dx"]), 0)
    assert np.isclose(_grid_index(-6.0 - tiny, 0.4, mesh["x0"], mesh["dx"]), 0)

    # At right-most grid (within tolerance), going left
    assert np.isclose(_grid_index(6.0 + tiny, -0.4, mesh["x0"], mesh["dx"]), 5)
    assert np.isclose(_grid_index(6.0 - tiny, -0.4, mesh["x0"], mesh["dx"]), 5)


def test__grid_distance():
    # Inside bin, going right
    assert np.isclose(_grid_distance(-3.2, 0.4, mesh["x0"], mesh["dx"]), 1.2 / 0.4)
    assert np.isclose(_grid_distance(3.2, 0.4, mesh["x0"], mesh["dx"]), 0.8 / 0.4)

    # Inside bin, going left
    assert np.isclose(_grid_distance(-3.2, -0.4, mesh["x0"], mesh["dx"]), 0.8 / 0.4)
    assert np.isclose(_grid_distance(3.2, -0.4, mesh["x0"], mesh["dx"]), 1.2 / 0.4)

    # Outside, moving closer
    assert np.isclose(_grid_distance(8.0, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4)
    assert np.isclose(_grid_distance(-8.0, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4)

    # At internal grid, going right
    assert np.isclose(_grid_distance(2.0, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4)
    assert np.isclose(_grid_distance(-2.0, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4)

    # At internal grid, going left
    assert np.isclose(_grid_distance(2.0, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4)
    assert np.isclose(_grid_distance(-2.0, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4)

    # At left-most grid, going right
    assert np.isclose(_grid_distance(-6.0, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4)

    # At right-most grid, going left
    assert np.isclose(_grid_distance(6.0, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4)

    # At internal grid (within tolerance), going right
    assert np.isclose(
        _grid_distance(2.0 + tiny, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )
    assert np.isclose(
        _grid_distance(2.0 - tiny, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )
    assert np.isclose(
        _grid_distance(-2.0 + tiny, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )
    assert np.isclose(
        _grid_distance(-2.0 - tiny, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )

    # At internal grid (within tolerance), going left
    assert np.isclose(
        _grid_distance(2.0 + tiny, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )
    assert np.isclose(
        _grid_distance(2.0 - tiny, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )
    assert np.isclose(
        _grid_distance(-2.0 + tiny, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )
    assert np.isclose(
        _grid_distance(-2.0 - tiny, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )

    # At left-most grid (within tolerance), going right
    assert np.isclose(
        _grid_distance(-6.0 + tiny, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )
    assert np.isclose(
        _grid_distance(-6.0 - tiny, 0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )

    # At right-most grid (within tolerance), going left
    assert np.isclose(
        _grid_distance(6.0 + tiny, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )
    assert np.isclose(
        _grid_distance(6.0 - tiny, -0.4, mesh["x0"], mesh["dx"]), 2.0 / 0.4
    )
