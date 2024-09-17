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


def particle(z, uz):
    u_ = np.sqrt(0.5 * (1.0 - uz**2))
    return {"x": -12.0, "y": 24.0, "z": z, "ux": u_, "uy": -u_, "uz": uz}


def surface(x, moving=False):
    return {"type": SURFACE_LINEAR + SURFACE_PLANE_Z, "J": -x, "moving": moving}


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
    # Moving
    # =================================================================================
    # TODO


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
    # TODO


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
    # TODO


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
