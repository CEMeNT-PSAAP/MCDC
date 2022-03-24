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
m = mcdc.material(capture=np.array([0.2]), scatter=np.array([[0.8]]))
m_barrier = mcdc.material(capture=np.array([1.0]), scatter=np.array([[4.0]]))

# Set surfaces
sx1 = mcdc.surface('plane-x', x=0.0,  bc="reflective")
sx2 = mcdc.surface('plane-x', x=10.0)
sx3 = mcdc.surface('plane-x', x=12.0)
sx4 = mcdc.surface('plane-x', x=20.0, bc="vacuum")
sy1 = mcdc.surface('plane-y', y=0.0,  bc="vacuum")
sy2 = mcdc.surface('plane-y', y=10.0)
sy3 = mcdc.surface('plane-y', y=20.0, bc="reflective")

# Set cells
mcdc.cell([+sx1, -sx2, +sy1, -sy2], m)
mcdc.cell([+sx3, -sx4, +sy1, -sy2], m)
mcdc.cell([+sx1, -sx4, +sy2, -sy3], m)
mcdc.cell([+sx2, -sx3, +sy1, -sy2], m_barrier)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(x=[0.0, 5.0], y=[0.0, 5.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally(scores=['flux'], x=np.linspace(0.0, 20.0, 41), 
                            y=np.linspace(0.0, 20.0, 41))

# Setting
mcdc.setting(N_hist=1E3, implicit_capture=True)
mcdc.run()
