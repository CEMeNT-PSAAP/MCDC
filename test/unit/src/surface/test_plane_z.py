import numpy as np

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
    SURFACE_LINEAR,
    SURFACE_PLANE_Z,
)

import mcdc.src.surface.common as common

from mcdc.src.surface.plane_z import (
    evaluate,
    reflect,
    get_normal_component,
    get_distance,
)


def particle(z, uz, t=0.0):
    u_ = np.sqrt(0.5 * (1.0 - uz**2))
    return {"x": -12.0, "y": 24.0, "z": z, "t": t, "ux": u_, "uy": -u_, "uz": uz}


def surface(z):
    return {"type": SURFACE_LINEAR + SURFACE_PLANE_Z, "J": -z, "moving": False}


tiny = COINCIDENCE_TOLERANCE * 0.8


# =====================================================================================
# Basics
# =====================================================================================


def test_evaluate():
    # Positive side
    result = evaluate(particle(3.0, -0.4), surface(1.0))
    assert np.isclose(result, 2.0)
    # Negative side
    result = evaluate(particle(5.0, 0.2), surface(9.0))
    assert np.isclose(result, -4.0)


def test_reflect():
    # From positive direction
    P = particle(3.0, 0.2)
    reflect(P, surface(123.0))
    assert np.isclose(P["uz"], -0.2)
    # From negative direction
    P = particle(4.0, -0.1)
    reflect(P, surface(-23.0))
    assert np.isclose(P["uz"], 0.1)


def test_get_normal_component():
    # Positive direction
    result = get_normal_component(particle(3.0, 0.4), surface(1.0))
    assert np.isclose(result, 0.4)
    # Negative direction
    result = get_normal_component(particle(5.0, -0.2), surface(9.0))
    assert np.isclose(result, -0.2)


def test_get_distance():
    # Positive side, moving closer
    result = get_distance(particle(3.0, -0.4), surface(1.0))
    assert np.isclose(result, 5.0)
    # Positive side, moving away
    result = get_distance(particle(4.0, 0.3), surface(1.0))
    assert np.isclose(result, INF)

    # Negative side, moving closer
    result = get_distance(particle(-3.0, 0.4), surface(1.0))
    assert np.isclose(result, 10.0)
    # Negative side, moving away
    result = get_distance(particle(-4.0, -0.3), surface(1.0))
    assert np.isclose(result, INF)

    # Positive side, parallel
    result = get_distance(particle(4.0, 0.0), surface(1.0))
    assert np.isclose(result, INF)
    # Positive side, parallel
    result = get_distance(particle(-4.0, 0.0), surface(1.0))
    assert np.isclose(result, INF)

    # At surface on the positive side, moving away
    result = get_distance(particle(1.0 + tiny, 0.4), surface(1.0))
    assert np.isclose(result, INF)
    # At surface on the positive side, moving closer
    result = get_distance(particle(1.0 + tiny, -0.4), surface(1.0))
    assert np.isclose(result, INF)

    # At surface on the negative side, moving away
    result = get_distance(particle(1.0 - tiny, -0.4), surface(1.0))
    assert np.isclose(result, INF)
    # At surface on the negative side, moving closer
    result = get_distance(particle(1.0 - tiny, 0.4), surface(1.0))
    assert np.isclose(result, INF)


# =====================================================================================
# Integrated
# =====================================================================================

Z = 10.0
time_grid = np.array([0.0, 5.0, 10.0, 15.0, INF])
N_move = len(time_grid) - 1
velocities = np.zeros((N_move, 3))
velocities[0, 2] = -1.0
velocities[1, 2] = 2.0
velocities[2, 2] = -3.0
translations = np.zeros((N_move + 1, 3))
for i in range(N_move):
    translations[i + 1] = translations[i] + velocities[i] * (
        time_grid[i + 1] - time_grid[i]
    )

moving_surface = {
    "type": SURFACE_LINEAR + SURFACE_PLANE_Z,
    "J": -Z,
    "moving": True,
    "N_move": N_move,
    "move_time_grid": time_grid,
    "move_translations": translations,
    "move_velocities": velocities,
}


def test_common_reflect():
    # Copied from test_reflect

    # From positive direction
    P = particle(3.0, 0.2)
    common.reflect(P, surface(123.0))
    assert np.isclose(P["uz"], -0.2)
    # From negative direction
    P = particle(4.0, -0.1)
    common.reflect(P, surface(-23.0))
    assert np.isclose(P["uz"], 0.1)


def test_common_evaluate():
    # =================================================================================
    # Static
    # =================================================================================
    # Copied from test_evaluate

    # Positive side
    result = common.evaluate(particle(3.0, -0.4), surface(1.0))
    assert np.isclose(result, 2.0)
    # Negative side
    result = common.evaluate(particle(5.0, 0.2), surface(9.0))
    assert np.isclose(result, -4.0)

    # =================================================================================
    # Moving
    # =================================================================================

    # First bin, positive side
    result = common.evaluate(particle(10.0, -0.4, 3.0), moving_surface)
    assert np.isclose(result, 3.0)
    # First bin, negative side
    result = common.evaluate(particle(1.0, 0.2, 3.0), moving_surface)
    assert np.isclose(result, -6.0)

    # First bin, at grid, positive side
    result = common.evaluate(particle(10.0, -0.4, 5.0), moving_surface)
    assert np.isclose(result, 5.0)
    # First bin, at grid, negative side
    result = common.evaluate(particle(1.0, 0.2, 5.0), moving_surface)
    assert np.isclose(result, -4.0)

    # Middle bin, positive side
    result = common.evaluate(particle(10.0, -0.4, 12.0), moving_surface)
    assert np.isclose(result, 1.0)
    # Middle bin, negative side
    result = common.evaluate(particle(1.0, 0.2, 12.0), moving_surface)
    assert np.isclose(result, -8.0)

    # Middle bin, at grid, positive side
    result = common.evaluate(particle(10.0, -0.4, 15.0), moving_surface)
    assert np.isclose(result, 10.0)
    # Middle bin, at grid, negative side
    result = common.evaluate(particle(-5.0, 0.2, 15.0), moving_surface)
    assert np.isclose(result, -5.0)

    # Last bin, positive side
    result = common.evaluate(particle(10.0, -0.4, 17.0), moving_surface)
    assert np.isclose(result, 10.0)
    # Last bin, negative side
    result = common.evaluate(particle(-5.0, 0.2, 18.0), moving_surface)
    assert np.isclose(result, -5.0)


def test_common_get_normal_component():
    # =================================================================================
    # Static
    # =================================================================================
    # Copied from test_get_normal_component

    # Positive direction
    result = common.get_normal_component(particle(3.0, 0.4), 1.0, surface(1.0))
    assert np.isclose(result, 0.4)
    # Negative direction
    result = common.get_normal_component(particle(5.0, -0.2), 1.0, surface(9.0))
    assert np.isclose(result, -0.2)

    # =================================================================================
    # Moving
    # =================================================================================

    # First move bin, same direction, faster
    result = common.get_normal_component(particle(3.0, -0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, -0.2 / 3.0)
    # First move bin, same direction, slower
    result = common.get_normal_component(particle(3.0, -0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, 0.2 / 2.0)
    # First move bin, same direction, same speed
    result = common.get_normal_component(particle(3.0, -0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, 0.0)

    # First move bin, opposite direction, faster
    result = common.get_normal_component(particle(3.0, 0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, 2.2 / 3.0)
    # First move bin, opposite direction, slower
    result = common.get_normal_component(particle(3.0, 0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, 1.8 / 2.0)
    # First move bin, opposite direction, same speed
    result = common.get_normal_component(particle(3.0, 0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, 2.0 * 0.4)

    # First move bin, at grid, same direction, faster
    result = common.get_normal_component(particle(3.0, 0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, 0.4 / 6.0)
    # First move bin, at grid, same direction, slower
    result = common.get_normal_component(particle(3.0, 0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, -1.2 / 2.0)
    # First move bin, at grid, same direction, same speed
    result = common.get_normal_component(particle(3.0, 0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, 0.0)

    # First move bin, at grid, opposite direction, faster
    result = common.get_normal_component(particle(3.0, -0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, -4.4 / 6.0)
    # First move bin, at grid, opposite direction, slower
    result = common.get_normal_component(particle(3.0, -0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, -2.8 / 2.0)
    # First move bin, at grid, opposite direction, same speed
    result = common.get_normal_component(particle(3.0, -0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, -4.0 / 5.0)

    # Middle move bin, same direction, faster
    result = common.get_normal_component(particle(3.0, 0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, 0.4 / 6.0)
    # Middle move bin, same direction, slower
    result = common.get_normal_component(particle(3.0, 0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, -1.2 / 2.0)
    # Middle move bin, same direction, same speed
    result = common.get_normal_component(particle(3.0, 0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, 0.0)

    # Middle move bin, opposite direction, faster
    result = common.get_normal_component(particle(3.0, -0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, -4.4 / 6.0)
    # Middle move bin, opposite direction, slower
    result = common.get_normal_component(particle(3.0, -0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, -2.8 / 2.0)
    # Middle move bin, opposite direction, same speed
    result = common.get_normal_component(particle(3.0, -0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, -4.0 / 5.0)

    # Middle move bin, at grid, same direction, faster
    result = common.get_normal_component(particle(3.0, -0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, -0.2 / 8.0)
    # Middle move bin, at grid, same direction, slower
    result = common.get_normal_component(particle(3.0, -0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, 2.2 / 2.0)
    # Middle move bin, at grid, same direction, same speed
    result = common.get_normal_component(particle(3.0, -0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, 0.0)

    # Middle move bin, at grid, opposite direction, faster
    result = common.get_normal_component(particle(3.0, 0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, 6.2 / 8.0)
    # Middle move bin, at grid, opposite direction, slower
    result = common.get_normal_component(particle(3.0, 0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, 3.8 / 2.0)
    # Middle move bin, at grid, opposite direction, same speed
    result = common.get_normal_component(particle(3.0, 0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, 6.0 / 7.5)

    # Last move bin, positive direction
    result = common.get_normal_component(particle(3.0, 0.4, 38.5), 6.0, moving_surface)
    assert np.isclose(result, 0.4)
    # Last move bin, negative direction
    result = common.get_normal_component(particle(3.0, -0.4, 38.5), 6.0, moving_surface)
    assert np.isclose(result, -0.4)


def test_check_sense():
    # =================================================================================
    # Static
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(3.0, -0.4), 1.0, surface(1.0))
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, 0.2), 1.0, surface(9.0))
    assert np.isclose(result, False)

    # At surface, positive side, positive direction
    result = common.check_sense(particle(1.0 + tiny, 0.4), 1.0, surface(1.0))
    assert np.isclose(result, True)
    # At surface, positive side, negative direction
    result = common.check_sense(particle(1.0 + tiny, -0.4), 1.0, surface(1.0))
    assert np.isclose(result, False)
    # At surface, negative side, positive direction
    result = common.check_sense(particle(9.0 - tiny, 0.2), 1.0, surface(9.0))
    assert np.isclose(result, True)
    # At surface, negative side, negative direction
    result = common.check_sense(particle(9.0 - tiny, -0.2), 1.0, surface(9.0))
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: first move bin, same direction, faster
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, -0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, -0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(7.5 + tiny, -0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(7.5 - tiny, -0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: first move bin, same direction, slower
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, -0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, -0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(7.5 + tiny, -0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(7.5 - tiny, -0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: first move bin, same direction, same speed
    #     At surface, this is undefined, but we choose to return false
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, -0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, -0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(7.5 + tiny, -0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(7.5 - tiny, -0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: first move bin, opposite direction, faster
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, 0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, 0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(7.5 + tiny, 0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(7.5 - tiny, 0.4, 2.5), 3.0, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: first move bin, opposite direction, slower
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, 0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, 0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(7.5 + tiny, 0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(7.5 - tiny, 0.4, 2.5), 2.0, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: first move bin, opposite direction, same speed
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, 0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, 0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(7.5 + tiny, 0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(7.5 - tiny, 0.4, 2.5), 2.5, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: first move bin, at grid, same direction, faster
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, 0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, 0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(5.0 + tiny, 0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(5.0 - tiny, 0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: first move bin, at grid, same direction, slower
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, 0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, 0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(5.0 + tiny, 0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(5.0 - tiny, 0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: first move bin, at grid, same direction, same speed
    #     At surface, this is undefined, but we choose to return false
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, 0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, 0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(5.0 + tiny, 0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(5.0 - tiny, 0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: first move bin, at grid, opposite direction, faster
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, -0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, -0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(5.0 + tiny, -0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(5.0 - tiny, -0.4, 5.0), 6.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: first move bin, at grid, opposite direction, slower
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, -0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, -0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(5.0 + tiny, -0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(5.0 - tiny, -0.4, 5.0), 2.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: first move bin, at grid, opposite direction, same speed
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(8.0, -0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, -0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(5.0 + tiny, -0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(5.0 - tiny, -0.4, 5.0), 5.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: middle move bin, same direction, faster
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(14.0, 0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, 0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(12.0 + tiny, 0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(12.0 - tiny, 0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: middle move bin, same direction, slower
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(14.0, 0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, 0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(12.0 + tiny, 0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(12.0 - tiny, 0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: middle move bin, same direction, same speed
    #     At surface, this is undefined, but we choose to return false
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(14.0, 0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, 0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(12.0 + tiny, 0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(12.0 - tiny, 0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: middle move bin, opposite direction, faster
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(14.0, -0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, -0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(12.0 + tiny, -0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(12.0 - tiny, -0.4, 8.5), 6.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: middle move bin, opposite direction, slower
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(14.0, -0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, -0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(12.0 + tiny, -0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(12.0 - tiny, -0.4, 8.5), 2.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: middle move bin, opposite direction, same speed
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(14.0, -0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(5.0, -0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(12.0 + tiny, -0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(12.0 - tiny, -0.4, 8.5), 5.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: middle move bin, at grid, same direction, faster
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(18.0, -0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, -0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(15.0 + tiny, -0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(15.0 - tiny, -0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: middle move bin, at grid, same direction, slower
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(18.0, -0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, -0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(15.0 + tiny, -0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(15.0 - tiny, -0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: middle move bin, at grid, same direction, same speed
    #     At surface, this is undefined, but we choose to return false
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(18.0, -0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(13.0, -0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(15.0 + tiny, -0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, False)
    # At surface, negative side
    result = common.check_sense(particle(15.0 - tiny, -0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, False)

    # =================================================================================
    # Moving: middle move bin, at grid, opposite direction, faster
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(18.0, 0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, 0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(15.0 + tiny, 0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(15.0 - tiny, 0.4, 10.0), 8.0, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: middle move bin, at grid, opposite direction, slower
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(18.0, 0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, 0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(15.0 + tiny, 0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(15.0 - tiny, 0.4, 10.0), 2.0, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: middle move bin, at grid, opposite direction, same speed
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(18.0, 0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(3.0, 0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side
    result = common.check_sense(particle(15.0 + tiny, 0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side
    result = common.check_sense(particle(15.0 - tiny, 0.4, 10.0), 7.5, moving_surface)
    assert np.isclose(result, True)

    # =================================================================================
    # Moving: last move bin
    # =================================================================================

    # Positive side
    result = common.check_sense(particle(14.0, -0.4, 18.5), 8.0, moving_surface)
    assert np.isclose(result, True)
    # Negative side
    result = common.check_sense(particle(-5.0, 0.4, 18.5), 2.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, positive side, positive direction
    result = common.check_sense(particle(tiny, 0.4, 18.5), 6.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, positive side, negative firection
    result = common.check_sense(particle(tiny, -0.4, 18.5), 6.0, moving_surface)
    assert np.isclose(result, False)

    # At surface, negative side, positive direction
    result = common.check_sense(particle(-tiny, 0.4, 18.5), 6.0, moving_surface)
    assert np.isclose(result, True)
    # At surface, negative side, negative direction
    result = common.check_sense(particle(-tiny, -0.4, 18.5), 6.0, moving_surface)
    assert np.isclose(result, False)


def test_common_get_distance():
    # =================================================================================
    # Static
    # =================================================================================
    # Copied from test_get_distance

    # Positive side, moving closer
    result = common.get_distance(particle(3.0, -0.4), 1.0, surface(1.0))
    assert np.isclose(result, 5.0)
    # Positive side, moving away
    result = common.get_distance(particle(4.0, 0.3), 1.0, surface(1.0))
    assert np.isclose(result, INF)

    # Negative side, moving closer
    result = common.get_distance(particle(-3.0, 0.4), 1.0, surface(1.0))
    assert np.isclose(result, 10.0)
    # Negative side, moving away
    result = common.get_distance(particle(-4.0, -0.3), 1.0, surface(1.0))
    assert np.isclose(result, INF)

    # Positive side, parallel
    result = common.get_distance(particle(4.0, 0.0), 1.0, surface(1.0))
    assert np.isclose(result, INF)
    # Positive side, parallel
    result = common.get_distance(particle(-4.0, 0.0), 1.0, surface(1.0))
    assert np.isclose(result, INF)

    # At surface on the positive side, moving away
    result = common.get_distance(particle(1.0 + tiny, 0.4), 1.0, surface(1.0))
    assert np.isclose(result, INF)
    # At surface on the positive side, moving closer
    result = common.get_distance(particle(1.0 + tiny, -0.4), 1.0, surface(1.0))
    assert np.isclose(result, INF)

    # At surface on the negative side, moving away
    result = common.get_distance(particle(1.0 - tiny, -0.4), 1.0, surface(1.0))
    assert np.isclose(result, INF)
    # At surface on the negative side, moving closer
    result = common.get_distance(particle(1.0 - tiny, 0.4), 1.0, surface(1.0))
    assert np.isclose(result, INF)

    # =================================================================================
    # Moving
    # =================================================================================

    # TODO
