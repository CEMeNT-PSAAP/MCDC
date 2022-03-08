import numpy as np
import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set cells
# =============================================================================

# Set materials
M = mcdc.Material(capture=np.array([1.0/3.0]), scatter=np.array([[1.0/3.0]]),
                  fission=np.array([1.0/3.0]), nu_p=np.array([2.3]))

# Set surfaces
S0 = mcdc.SurfacePlaneX(-1E10, "reflective")
S1 = mcdc.SurfacePlaneX(1E10, "reflective")

# Set cells
C = mcdc.Cell([+S0, -S1], M)
cells = [C]

# =============================================================================
# Set source
# =============================================================================

position  = mcdc.DistPoint()
direction = mcdc.DistPointIsotropic()
time      = mcdc.DistDelta(0.0)

Src = mcdc.SourceSimple(position=position, direction=direction, time=time)
sources = [Src]

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

mcdc.set_problem(cells, sources, N_hist=1E3)
f = np.load('azurv1_pl.npz')
mcdc.set_tally(scores=['flux'], x=f['x'], t=f['t'])
# TODO: use time boundary instead
mcdc.set_population_control(census_time=np.array([20.0]))
mcdc.run()
