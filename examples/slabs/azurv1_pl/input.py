import numpy as np
import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set materials
# =============================================================================

SigmaC = np.array([1.0/3.0])
SigmaS = np.array([[1.0/3.0]])
SigmaF = np.array([[1.0/3.0]])
nu     = np.array([2.3])
M = mcdc.Material(SigmaC, SigmaS, SigmaF, nu)

# =============================================================================
# Set cells
# =============================================================================

# Set surfaces
S0 = mcdc.SurfacePlaneX(-1E10, "reflective")
S1 = mcdc.SurfacePlaneX(1E10, "reflective")

# Set cells
C = mcdc.Cell([+S0, -S1], M)
cells = [C]

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
time = mcdc.DistDelta(0.0)

# Create the source
Source = mcdc.SourceSimple(pos,dir,g,time,cell=C)

# =============================================================================
# Set filters and tallies
# =============================================================================

# Load grids
grid = np.load('azurv1_pl.npz')
time_filter = mcdc.FilterTime(grid['t'])
spatial_filter = mcdc.FilterPlaneX(grid['x'])

T = mcdc.Tally('tally', scores=['flux', 'flux-edge', 'flux-face'],
               spatial_filter=spatial_filter,
               time_filter=time_filter)

tallies = [T]

# =============================================================================
# Set and run simulator
# =============================================================================

# Set speed
speeds = np.array([1.0])

# Set simulator
simulator = mcdc.Simulator(speeds, cells, Source, tallies=tallies, 
                           N_hist=10000)

# Set population control and census
simulator.set_pct(census_time=np.array([20.0]))

# Run
simulator.run()
