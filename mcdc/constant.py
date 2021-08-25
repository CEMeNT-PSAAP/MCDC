import numpy as np


# RNG default parameters
LCG_G      = 2806196910506780709
LCG_C      = 1
LCG_MOD    = 2**63
LCG_STRIDE = 152917
LCG_SEED   = 1
LCG_G_B05  = 3512401965023503517 # [Brown 2005]
LCG_C_B05  = 0

# EVENT
EVENT_COLLISION = 1
EVENT_SURFACE   = 2
EVENT_CENSUS    = 3

# Misc.
SMALL      = 1E-10  # to ensure spatial grid crossing
VERY_SMALL = 1E-15  # to ensure time grid crossing
INF        = np.inf
PI         = np.pi
EPSILON    = 1E-14  # to avoid precision error in tally filters

