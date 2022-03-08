import numpy as np

# RNG default parameters
LCG_G      = 2806196910506780709
LCG_C      = 1
LCG_MOD    = 2**63
LCG_STRIDE = 152917
LCG_SEED   = 1
LCG_G_B05  = 3512401965023503517 # [Brown 2005]
LCG_C_B05  = 0

# Events
EVENT_COLLISION     = 1
EVENT_SURFACE       = 2
EVENT_CENSUS        = 3
EVENT_MESH          = 4
EVENT_SCATTERING    = 5
EVENT_FISSION       = 6
EVENT_CAPTURE       = 7
EVENT_TIME_REACTION = 8

# Misc.
INF       = np.inf
PI        = np.pi
EPSILON   = 1E-15 # To ensure non-zero value
PRECISION = 1E-10 # Used in surface crossing
