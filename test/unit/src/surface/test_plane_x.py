import numpy as np

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)

from mcdc.src.surface.plane_x import (
    evaluate,
    reflect,
    get_normal_component,
    get_distance,
)


def particle(x, ux, t):
    return {"x": x, "ux": ux, "t": t}


speed = 1.0


def surface(x):
    return {"J": -x, "moving": False}


tiny = COINCIDENCE_TOLERANCE * 0.8


def test_evaluate():
    # [Static] Positive side
    result = evaluate(particle(8.0, 1.0, 0.0), surface(5.0))
    assert np.isclose(result, 3.0)
    result = evaluate(particle(8.0, -1.0, 0.0), surface(5.0))
    assert np.isclose(result, 3.0)

    # [Static] Negative side
    result = evaluate(particle(-8.0, 1.0, 0.0), surface(5.0))
    assert np.isclose(result, -13.0)
    result = evaluate(particle(-8.0, -1.0, 0.0), surface(5.0))
    assert np.isclose(result, -13.0)

    # TODO: [Moving]


# TODO: reflect
# TODO: get_normal_component [Static and Moving]
# TODO: get_distance [Static and Moving]