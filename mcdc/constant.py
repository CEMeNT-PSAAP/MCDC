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
EVENT_MESH      = 4

# Physics
mass_n = 1.67492749804E-27 # kg

# Unit
eV_to_J       = 1.60218E-19
eV_to_speed_n = np.sqrt(eV_to_J*2/mass_n)*100.0 # cm/s

# Misc.
INF        = np.inf
PI         = np.pi
SMALL      = 1E-10  # to ensure spatial grid crossing
VERY_SMALL = 1E-15  # to ensure time grid crossing
EPSILON    = 1E-14  # to avoid precision error in tally filters
