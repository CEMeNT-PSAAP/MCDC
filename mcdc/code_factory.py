import numpy as np

from numba import cuda, njit

import mcdc.local as local
import mcdc.type_ as type_


# ==============================================================================
# Local array and object (see local.py)
# ==============================================================================


def make_locals(input_deck):
    # Hardware target
    target = input_deck.setting["target"]

    # Problem-dependent sizes
    G = input_deck.materials[0].G
    J = input_deck.materials[0].J
    N_RPN = max([np.sum(np.array(x._region_RPN) >= 0.0) for x in input_deck.cells])

    # Make the locals
    local.translation = local_array(type_.float64, 3, target)
    local.energy_group_array = local_array(type_.float64, G, target)
    local.precursor_group_array = local_array(type_.float64, J, target)
    local.RPN_array = local_array(type_.bool_, N_RPN, target)
    local.particle = local_object(type_.particle, target)
    local.particle_record = local_object(type_.particle_record, target)


def local_array(dtype, size, target):
    struct = type_.into_dtype([("values", dtype, (size,))])

    @njit
    def cpu():
        return np.zeros(1, dtype=struct)[0]

    @cuda.jit(device=True)
    def gpu():
        return cuda.local.array(1, dtype=struct)[0]

    return cpu if target == "cpu" else gpu


def local_object(dtype, target):

    @njit
    def cpu():
        return np.zeros(1, dtype=dtype)[0]

    @cuda.jit(device=True)
    def gpu():
        return cuda.local.array(1, dtype=dtype)[0]

    return cpu if target == "cpu" else gpu
