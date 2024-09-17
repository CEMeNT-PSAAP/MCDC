import numpy as np

from mcdc.constant import INF
from mcdc.src.surface.common import (
    _get_move_idx,
    _translate_particle_position,
    _translate_particle_direction,
)


def particle(x, y, z, t, ux, uy, uz):
    return {"x": x, "y": y, "z": z, "t": t, "ux": ux, "uy": uy, "uz": uz}


time_grid = np.array((0.0, 1.0, 3.0, 6.0, 10.0, INF))
N_move = len(time_grid) - 1
move_translations = np.zeros((N_move + 1, 3))
move_velocities = np.zeros((N_move, 3))

for i in range(N_move - 1):
    move_translations[i + 1][0] = 1.0 * (i + 1)
    move_translations[i + 1][1] = 2.0 * (i + 1)
    move_translations[i + 1][2] = 3.0 * (i + 1)
    move_velocities[i] = (move_translations[i + 1] - move_translations[i]) / (
        time_grid[i + 1] - time_grid[i]
    )
move_translations[-1] = move_translations[-2]

surface = {
    "N_move": N_move,
    "move_translations": move_translations,
    "move_velocities": move_velocities,
    "move_time_grid": time_grid,
}


def test_get_move_idx():
    # In bin
    result = _get_move_idx(0.4, surface)
    assert np.isclose(result, 0)
    result = _get_move_idx(5.4, surface)
    assert np.isclose(result, 2)
    result = _get_move_idx(15.0, surface)
    assert np.isclose(result, 4)

    # At grid
    result = _get_move_idx(0.0, surface)
    assert np.isclose(result, 0)
    result = _get_move_idx(1.0, surface)
    assert np.isclose(result, 1)
    result = _get_move_idx(3.0, surface)
    assert np.isclose(result, 2)
    result = _get_move_idx(6.0, surface)
    assert np.isclose(result, 3)
    result = _get_move_idx(10.0, surface)
    assert np.isclose(result, 4)


def test_translate_particle_position():
    u = (1.0 / 3.0) ** 0.5

    # At grid
    t = 6.0
    P = particle(5.0, 3.0, 8.0, t, u, -u, u)
    idx = _get_move_idx(t, surface)
    _translate_particle_position(P, surface, idx)
    x = 5.0 - move_translations[3][0]
    y = 3.0 - move_translations[3][1]
    z = 8.0 - move_translations[3][2]
    assert np.isclose(P["x"], x)
    assert np.isclose(P["y"], y)
    assert np.isclose(P["z"], z)
    assert np.isclose(P["t"], t)
    assert np.isclose(P["ux"], u)
    assert np.isclose(P["uy"], -u)
    assert np.isclose(P["uz"], u)

    # In bin
    t = 6.5
    P = particle(5.0, 3.0, 8.0, t, u, -u, u)
    idx = _get_move_idx(t, surface)
    _translate_particle_position(P, surface, idx)
    x = 5.0 - move_translations[3][0] - move_velocities[3][0] * (t - time_grid[3])
    y = 3.0 - move_translations[3][1] - move_velocities[3][1] * (t - time_grid[3])
    z = 8.0 - move_translations[3][2] - move_velocities[3][2] * (t - time_grid[3])
    assert np.isclose(P["x"], x)
    assert np.isclose(P["y"], y)
    assert np.isclose(P["z"], z)
    assert np.isclose(P["t"], t)
    assert np.isclose(P["ux"], u)
    assert np.isclose(P["uy"], -u)
    assert np.isclose(P["uz"], u)


def test_translate_particle_direction():
    speed = 2.0
    u = (1.0 / 3.0) ** 0.5

    # At grid
    t = 6.0
    P = particle(5.0, 3.0, 8.0, t, u, -u, u)
    idx = _get_move_idx(t, surface)
    _translate_particle_direction(P, speed, surface, idx)
    ux = u - move_velocities[3][0] / speed
    uy = -u - move_velocities[3][1] / speed
    uz = u - move_velocities[3][2] / speed
    assert np.isclose(P["x"], 5.0)
    assert np.isclose(P["y"], 3.0)
    assert np.isclose(P["z"], 8.0)
    assert np.isclose(P["t"], 6.0)
    assert np.isclose(P["ux"], ux)
    assert np.isclose(P["uy"], uy)
    assert np.isclose(P["uz"], uz)

    # At grid
    t = 6.5
    P = particle(5.0, 3.0, 8.0, t, u, -u, u)
    idx = _get_move_idx(t, surface)
    _translate_particle_direction(P, speed, surface, idx)
    ux = u - move_velocities[3][0] / speed
    uy = -u - move_velocities[3][1] / speed
    uz = u - move_velocities[3][2] / speed
    assert np.isclose(P["x"], 5.0)
    assert np.isclose(P["y"], 3.0)
    assert np.isclose(P["z"], 8.0)
    assert np.isclose(P["t"], 6.5)
    assert np.isclose(P["ux"], ux)
    assert np.isclose(P["uy"], uy)
    assert np.isclose(P["uz"], uz)
