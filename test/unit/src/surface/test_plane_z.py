import numpy as np

from mcdc.constant import (
    COINCIDENCE_TOLERANCE,
    INF,
)

from mcdc.src.surface.plane_z import (
    evaluate,
    reflect,
    get_normal_component,
    get_distance,
)


def particle(x, y, z, t, ux, uy, uz):
    return {"x": x, "y": y, "z": z, "t": t, "ux": ux, "uy": uy, "uz": uz}


speed = 2.0


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
