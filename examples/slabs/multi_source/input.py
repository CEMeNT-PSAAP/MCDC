import numpy as np
import sys, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set materials
# =============================================================================

M = mcdc.Material(capture=np.array([1.0]))

# =============================================================================
# Set cells
# =============================================================================

# Set surfaces
S0 = mcdc.SurfacePlaneX(0.0, "vacuum")
S1 = mcdc.SurfacePlaneX(6.0, "vacuum")

# Set cells
C = mcdc.Cell([+S0, -S1], M)
cells = [C]

# =============================================================================
# Set source
# =============================================================================

# Position distribution
pos1 = mcdc.DistPoint(mcdc.DistDelta(mcdc.constant.SMALL), mcdc.DistDelta(0.0), 
                             mcdc.DistDelta(0.0))
pos2 = mcdc.DistPoint(mcdc.DistUniform(4.0,5.0), mcdc.DistDelta(0.0), 
                             mcdc.DistDelta(0.0))

# Direction distribution
dir1 = mcdc.DistPoint(mcdc.DistDelta(1.0), mcdc.DistDelta(0.0), 
                             mcdc.DistDelta(0.0))
dir2 = mcdc.DistPointIsotropic()

# Energy group distribution
g = mcdc.DistDelta(0)

# Time distribution
time= mcdc.DistDelta(0.0)

# Create the source
Src1 = mcdc.SourceSimple(pos1,dir1,g,time,prob=0.4)
Src2 = mcdc.SourceSimple(pos2,dir2,g,time,prob=0.6)
sources = [Src1, Src2]

# =============================================================================
# Set filters and tallies
# =============================================================================

spatial_filter = mcdc.FilterPlaneX(np.linspace(0.0, 6.0, 61))

T = mcdc.Tally('tally', scores=['flux', 'flux-face', 'current', 'current-face'], 
               spatial_filter=spatial_filter)

tallies = [T]

# =============================================================================
# Set and run simulator
# =============================================================================

# Set simulator
simulator = mcdc.Simulator(cells=cells, sources=sources, tallies=tallies, 
                           N_hist=1E4)

# Run
simulator.run()
