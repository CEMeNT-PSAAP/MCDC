import numpy as np
import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set cells
# =============================================================================

# Set materials
M1 = mcdc.Material(capture=np.array([0.0]), scatter=np.array([[0.9]]),
                   fission=np.array([0.1]), nu_p=np.array([6.0]))
M2 = mcdc.Material(capture=np.array([0.68]), scatter=np.array([[0.2]]),
                   fission=np.array([0.12]), nu_p=np.array([2.5]))

# Set surfaces
S0 = mcdc.SurfacePlaneX(0.0, "vacuum")
S1 = mcdc.SurfacePlaneX(1.5)
S2 = mcdc.SurfacePlaneX(2.5, "vacuum")

# Set cells
C1 = mcdc.Cell([+S0, -S1], M1)
C2 = mcdc.Cell([+S1, -S2], M2)
cells = [C1, C2]

# =============================================================================
# Set source
# =============================================================================

position  = mcdc.DistPoint(mcdc.DistUniform(0.0, 2.5))
direction = mcdc.DistPointIsotropic()

Src = mcdc.SourceSimple(position=position, direction=direction)
sources = [Src]

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

mcdc.set_problem(cells, sources, N_hist=5E3)
mcdc.set_kmode(N_iter=40)
x_grid = np.array([0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 
                   1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5])
mcdc.set_tally(scores=['flux'], x=x_grid)
mcdc.run()
