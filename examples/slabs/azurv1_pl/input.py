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
# Set and run simulator
# =============================================================================

# Set simulator
simulator = mcdc.Simulator(cells=cells, sources=sources, N_hist=1E4)
f = np.load('azurv1_pl.npz')
simulator.set_tally(scores=['flux'], x=f['x'], t=f['t'])
f.close()

# Set population control and census
simulator.set_pct(census_time=np.array([20.0]))

# Run
simulator.run()
