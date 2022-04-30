import numpy as np

# Disable Numba-JIT for pure Python mode
from numba import config
config.DISABLE_JIT = True

# Get path to mcdc (not necessary if mcdc is installed)
import sys
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set model
# =============================================================================

# Set materials
m1 = mcdc.material(capture=np.array([0.0]), scatter=np.array([[0.9]]),
                   fission=np.array([0.1]), nu_p=np.array([6.0]))
m2 = mcdc.material(capture=np.array([0.68]), scatter=np.array([[0.2]]),
                   fission=np.array([0.12]), nu_p=np.array([2.5]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
s2 = mcdc.surface('plane-x', x=1.5)
s3 = mcdc.surface('plane-x', x=2.5, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m1)
mcdc.cell([+s2, -s3], m2)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(x=[0.0, 2.5], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
mcdc.tally(scores=['flux'], x=[0.0, 2.5, 50])

# Setting
mcdc.setting(N_hist=5E3)
mcdc.eigenmode(N_iter=40, alpha_mode=True)

# Run
mcdc.run()
