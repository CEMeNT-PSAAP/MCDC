import numpy as np
import numba as nb
import sys

float64 = np.float64
int64   = np.int64
uint64  = np.uint64
bool_   = np.bool_

# ==============================================================================
# Particle
# ==============================================================================

particle = np.dtype([('x', float64), ('y', float64), ('z', float64),
                     ('ux', float64), ('uy', float64), ('uz', float64),
                     ('time', float64), ('speed', float64), ('group', uint64),
                     ('weight', float64), ('alive', bool_), ('cell_ID', int64), 
                     ('surface_ID', int64)])

@nb.njit
def make_particle():
    P = np.zeros(1, dtype=particle)[0]
    P['x']          = 0.0
    P['y']          = 0.0
    P['z']          = 0.0
    P['ux']         = 0.0
    P['uy']         = 0.0
    P['uz']         = 0.0
    P['group']      = 0
    P['time']       = 0.0
    P['weight']     = 1.0
    P['alive']      = True
    P['speed']      = 1.0
    P['cell_ID']    = -1
    P['surface_ID'] = -1
    return P

# ==============================================================================
# Particle bank
# ==============================================================================

def make_bank(tag, max_size):
    type_bank = np.dtype([('tag', 'U10'), ('particles', particle, (max_size,)),
        ('max_size', uint64), ('size', uint64)])
    bank = np.array([(tag, np.full(max_size, make_particle()), max_size, 0)], 
                    dtype=type_bank)[0]
    return bank
