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
m      = mcdc.material(capture=np.array([0.05]), scatter=np.array([[0.05]]))
m_void = mcdc.material(capture=np.array([5E-5]), scatter=np.array([[5E-5]]))

# Set surfaces
sx1 = mcdc.surface('plane-x', x=0.0,  bc="reflective")
sx2 = mcdc.surface('plane-x', x=10.0)
sx3 = mcdc.surface('plane-x', x=50.0)
sx4 = mcdc.surface('plane-x', x=100.0, bc="vacuum")
sy1 = mcdc.surface('plane-y', y=0.0,  bc="reflective")
sy2 = mcdc.surface('plane-y', y=10.0)
sy3 = mcdc.surface('plane-y', y=50.0)
sy4 = mcdc.surface('plane-y', y=100.0, bc="vacuum")
sz1 = mcdc.surface('plane-z', z=0.0,  bc="reflective")
sz2 = mcdc.surface('plane-z', z=10.0)
sz3 = mcdc.surface('plane-z', z=50.0)
sz4 = mcdc.surface('plane-z', z=100.0, bc="vacuum")

# Set cells
## Soruce
mcdc.cell([+sx1, -sx2, +sy1, -sy2, +sz1, -sz2], m)
## Voids
mcdc.cell([+sx1, -sx3, +sy2, -sy3, +sz1, -sz3], m_void)
mcdc.cell([+sx2, -sx3, +sy1, -sy2, +sz1, -sz3], m_void)
mcdc.cell([+sx1, -sx2, +sy1, -sy2, +sz2, -sz3], m_void)
## Shield
mcdc.cell([+sx1, -sx4, +sy3, -sy4, +sz1, -sz4], m)
mcdc.cell([+sx3, -sx4, +sy1, -sy3, +sz1, -sz4], m)
mcdc.cell([+sx1, -sx3, +sy1, -sy3, +sz3, -sz4], m)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(x=[0.0, 10.0], y=[0.0, 10.0], z=[0.0, 10.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally(scores=['flux'], x=np.linspace(0.0, 100.0, 11), 
                            y=np.linspace(0.0, 100.0, 11),
                            z=np.linspace(0.0, 100.0, 11))

# Setting
mcdc.setting(N_hist=1E4)
mcdc.run()
