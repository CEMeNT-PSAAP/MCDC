import numpy as np

from mcdc import type_
from mcdc.kernel import surface_evaluate
from mcdc.constant import INF


# =============================================================================
# Static surface
# =============================================================================


def tmp_plane_x():
    type_.make_type_surface(1)
    P = np.zeros(1, dtype=type_.particle)[0]
    S = np.zeros(1, dtype=type_.surface)[0]

    x = 5.0
    trans = np.array([0.0, 0.0, 0.0])

    S["G"] = 1.0
    S["linear"] = True
    S["J"] = np.array([-x, -x])
    S["t"] = np.array([0.0, INF])

    # Surface on the left
    P["x"] = 4.0
    result = surface_evaluate(P, S, trans)
    assert result < 0.0

    # Surface on the right
    P["x"] = 9.0
    result = surface_evaluate(P, S, trans)
    assert result > 0.0
