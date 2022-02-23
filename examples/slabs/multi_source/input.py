import numpy as np
import sys, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set cells
# =============================================================================

# Set materials
M = mcdc.Material(capture=np.array([0.5]), scatter=np.array([[1.0]]))

# Set surfaces
S0 = mcdc.SurfacePlaneX(0.0, "vacuum")
S1 = mcdc.SurfacePlaneX(6.0, "vacuum")

# Set cells
C = mcdc.Cell([+S0, -S1], M)
cells = [C]

# =============================================================================
# Set sources
# =============================================================================
# Two sources with different probabilities

position1  = mcdc.DistPoint(x=mcdc.DistDelta(1E-10))
direction1 = mcdc.DistPoint(x=mcdc.DistDelta(1.0))
Src1 = mcdc.SourceSimple(position=position1, direction=direction1, prob=0.4)

position2  = mcdc.DistPoint(x=mcdc.DistUniform(4.0,5.0))
direction2 = mcdc.DistPointIsotropic()
Src2 = mcdc.SourceSimple(position=position2, direction=direction2, prob=0.6)

sources = [Src1, Src2]

# =============================================================================
# Set and run simulator
# =============================================================================

# Set simulator and tally
simulator = mcdc.Simulator(cells=cells, sources=sources, N_hist=1E4)
simulator.set_tally(scores=['flux', 'current'], x=np.linspace(0.0, 6.0, 61))

# Run
simulator.run()
