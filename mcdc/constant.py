import numpy as np


# Misc.
SMALL_KICK = 1E-10  # small kick to ensure surface crossing
INF        = np.inf
PI         = np.pi

# RNG default parameters
LCG_G      = 2806196910506780709
LCG_C      = 1
LCG_MOD    = 2**63
LCG_STRIDE = 152917
LCG_SEED   = 1
LCG_G_B05  = 3512401965023503517 # [Brown 2005]
LCG_C_B05  = 0
