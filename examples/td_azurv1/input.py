import numpy as np
import sys, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../')

import mcdc


# =============================================================================
# Set material XS
# =============================================================================

speeds = [1.0]

SigmaT = np.array([1.0])
nu     = np.array([0.0])
SigmaF = np.array([[0.0]])
SigmaS = np.array([[[0.98]]])
M1 = mcdc.Material(SigmaT,SigmaS,nu,SigmaF)

# =============================================================================
# Set cells
# =============================================================================

# Set surfaces
S0 = mcdc.SurfacePlaneX(-15.,"vacuum")
S1 = mcdc.SurfacePlaneX(15.,"vacuum")

# Set cells
C1 = mcdc.Cell([[S0,+1],[S1,-1]],M1)
cells = [C1]

# =============================================================================
# Set source
# =============================================================================

# Position distribution
pos = mcdc.DistPoint(mcdc.DistDelta(0.0), mcdc.DistDelta(0.0), 
                             mcdc.DistDelta(0.0))
# Direction distribution
dir = mcdc.DistPointIsotropic()

# Energy group distribution
g = mcdc.DistDelta(0)

# Time distribution
time= mcdc.DistDelta(0.0)

# Create the source
Source = mcdc.SourceSimple(pos,dir,g,time,cell=C1)

# =============================================================================
# Set filters and tallies
# =============================================================================

# Load grids
grid = np.load('azurv1_pl.npz')
time_filter = mcdc.FilterTime(grid['t'])
spatial_filter = mcdc.FilterPlaneX(grid['x'])

T = mcdc.Tally('tally', scores=['flux-edge','flux-face','flux'],
               spatial_filter=spatial_filter,
               time_filter=time_filter)

tallies = [T]

# =============================================================================
# Set and run simulator (for each value of N)
# =============================================================================

# Set simulator
simulator = mcdc.Simulator(speeds, cells, Source, tallies=tallies, 
                           N_hist=1000)

# Set VRT
#simulator.set_vrt(continuous_capture=False, implicit_fission=True,
#                  wgt_roulette=0.25)

# Run
simulator.run()
