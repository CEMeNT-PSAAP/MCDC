import numpy as np
import sys, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set materials
# =============================================================================

M1 = mcdc.Material(capture=np.array([1.0]))
M2 = mcdc.Material(capture=np.array([1.5]))
M3 = mcdc.Material(capture=np.array([2.0]))

# =============================================================================
# Set cells
# =============================================================================

# Set surfaces
S0 = mcdc.SurfacePlaneX(0.0, "vacuum")
S1 = mcdc.SurfacePlaneX(2.0, "transmission")
S2 = mcdc.SurfacePlaneX(4.0, "transmission")
S3 = mcdc.SurfacePlaneX(6.0, "vacuum")

# Set cells
C1 = mcdc.Cell([+S0, -S1], M2)
C2 = mcdc.Cell([+S1, -S2], M3)
C3 = mcdc.Cell([+S2, -S3], M1)
cells = [C1, C2, C3]

# =============================================================================
# Set source
# =============================================================================

# Position distribution
pos = mcdc.DistPoint(mcdc.DistUniform(0.0,6.0), mcdc.DistDelta(0.0), 
                             mcdc.DistDelta(0.0))
# Direction distribution
dir = mcdc.DistPointIsotropic()

# Energy group distribution
g = mcdc.DistDelta(0)

# Time distribution
time= mcdc.DistDelta(0.0)

# Create the source
Src = mcdc.SourceSimple(pos,dir,g,time)
sources = [Src]

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
