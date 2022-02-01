import numpy as np
import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc


# =============================================================================
# Set cells
# =============================================================================

# Set materials
M1 = mcdc.Material(capture=np.array([1.0]))
M2 = mcdc.Material(capture=np.array([1.5]))
M3 = mcdc.Material(capture=np.array([2.0]))

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
# Set sources
# =============================================================================
# Parallel beam, at x=0, going in the +x direction
# Source needs to be slightly shifted to the right to make sure it starts in
# the right cell.

position = mcdc.DistPoint(mcdc.DistDelta(1E-10), mcdc.DistDelta(0.0), 
                          mcdc.DistDelta(0.0))

direction = mcdc.DistPoint(mcdc.DistDelta(1.0), mcdc.DistDelta(0.0), 
                           mcdc.DistDelta(0.0))

Src = mcdc.SourceSimple(position=position, direction=direction)

sources = [Src]

# =============================================================================
# Set filters and tallies
# =============================================================================

spatial_filter = mcdc.FilterPlaneX(np.linspace(0.0, 6.0, 61))

T = mcdc.Tally('tally', scores=['flux', 'flux-face'], 
               spatial_filter=spatial_filter)

tallies = [T]

# =============================================================================
# Set and run simulator
# =============================================================================

# Set simulator
simulator = mcdc.Simulator(cells=cells, sources=sources, tallies=tallies, 
                           N_hist=1E4)

window = np.load('phi.npy')

simulator.set_weight_window(x=np.linspace(0.0, 6.0, 61),
                            window=window)

# Run
simulator.run()
