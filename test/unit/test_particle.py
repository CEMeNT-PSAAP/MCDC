import numpy as np

import mcdc.numba_types as type_
from mcdc.transport.particle import copy


def test_copy_particle_data_to_active_particle():
    source = np.zeros(1, dtype=type_.particle_data)
    target = np.zeros(1, dtype=type_.particle)

    values = {
        "x": 1.0,
        "y": 2.0,
        "z": 3.0,
        "t": 4.0,
        "ux": 0.1,
        "uy": 0.2,
        "uz": 0.3,
        "group": 7,
        "E": 8.0,
        "w": 9.0,
        "particle_type": 10,
        "rng_seed": 11,
    }
    for field, value in values.items():
        source[0][field] = value

    copy(target, source)

    for field, value in values.items():
        assert target[0][field] == value
