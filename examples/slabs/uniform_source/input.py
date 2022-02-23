import numpy as np
import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set cells
# =============================================================================
# Three slab layers with different materials

# Set materials
M1 = mcdc.Material(capture=np.array([1.0]))
M2 = mcdc.Material(capture=np.array([1.5]))
M3 = mcdc.Material(capture=np.array([2.0]))

# Set surfaces
S0 = mcdc.SurfacePlaneX(0.0, "vacuum")
S1 = mcdc.SurfacePlaneX(2.0)
S2 = mcdc.SurfacePlaneX(4.0)
S3 = mcdc.SurfacePlaneX(6.0, "vacuum")

# Set cells
C1 = mcdc.Cell([+S0, -S1], M2)
C2 = mcdc.Cell([+S1, -S2], M3)
C3 = mcdc.Cell([+S2, -S3], M1)
cells = [C1, C2, C3]

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the medium

position  = mcdc.DistPoint(x=mcdc.DistUniform(0.0,6.0))
direction = mcdc.DistPointIsotropic()

Src = mcdc.SourceSimple(position=position, direction=direction)
sources = [Src]

# =============================================================================
# Set and run simulator
# =============================================================================

# Set simulator and tally
simulator = mcdc.Simulator(cells=cells, sources=sources, N_hist=1E4)
simulator.set_tally(scores=['flux', 'current'], x=np.linspace(0.0, 6.0, 61))

# Run
simulator.run()
