import numpy as np
import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set cells
# =============================================================================

# Set materials
m = mcdc.material(capture=np.array([1.0/3.0]), scatter=np.array([[1.0/3.0]]),
                  fission=np.array([1.0/3.0]), nu_p=np.array([2.3]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=-1E10, bc="reflective")
s2 = mcdc.surface('plane-x', x=1E10,  bc="reflective")

# Set cells
c = mcdc.cell([+s1, -s2], m)
cells = [c]

# =============================================================================
# Set source
# =============================================================================

source = mcdc.source(point=[0.0,0.0,0.0], isotropic=True)

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

mcdc.set_problem(cells, source, N_hist=1E3)
f = np.load('azurv1_pl.npz')
mcdc.set_tally(scores=['flux'], x=f['x'], t=f['t'])
# TODO: use time boundary instead
mcdc.set_population_control(census_time=np.array([20.0]))
mcdc.run()
