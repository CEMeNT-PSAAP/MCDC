import numpy as np
import sys, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set materials
# =============================================================================

SigmaC = np.array([1.0/3.0])
SigmaS = np.array([[1.0/3.0]])
SigmaF = np.array([[1.0/3.0]])
nu     = np.array([2.0])
M1 = mcdc.Material(SigmaC, SigmaS, SigmaF, nu)

SigmaC = np.array([1.0/3.0])
SigmaS = np.array([[2.0/3.0]])
SigmaF = np.array([[0.0]])
nu     = np.array([0.0])
M2 = mcdc.Material(SigmaC, SigmaS, SigmaF, nu)

# =============================================================================
# Set cells
# =============================================================================

# Set surfaces
S0 = mcdc.SurfacePlaneX(0.0, "vacuum")
S1 = mcdc.SurfacePlaneX(2.0, "transmission")
S2 = mcdc.SurfacePlaneX(4.0, "transmission")
S3 = mcdc.SurfacePlaneX(5.0, "vacuum")

# Set cells
C1 = mcdc.Cell([+S0, -S1], M2)
C2 = mcdc.Cell([+S1, -S2], M1)
C3 = mcdc.Cell([+S2, -S3], M2)
cells = [C1, C2, C3]

# =============================================================================
# Set source
# =============================================================================

# Position distribution
pos = mcdc.DistPoint(mcdc.DistUniform(2.0,4.0), mcdc.DistDelta(0.0), 
                             mcdc.DistDelta(0.0))
# Direction distribution
dir = mcdc.DistPointIsotropic()

# Energy group distribution
g = mcdc.DistDelta(0)

# Time distribution
time= mcdc.DistDelta(0.0)

# Create the source
Source = mcdc.SourceSimple(pos,dir,g,time)

# =============================================================================
# Set filters and tallies
# =============================================================================

spatial_filter1 = mcdc.FilterPlaneX(np.linspace(0.0, 5.0, 51))
spatial_filter2 = mcdc.FilterSurface([S0, S3])

T1 = mcdc.Tally('tally1', scores=['flux', 'flux-face', 'current', 'current-face'], 
                spatial_filter=spatial_filter1)
T2 = mcdc.Tally('tally2', scores=['partial_crossing'], 
                spatial_filter=spatial_filter2)

tallies = [T1, T2]

# =============================================================================
# Set and run simulator
# =============================================================================

# Set speed
speeds = np.array([1.0])

# Set simulator
simulator = mcdc.Simulator(speeds, cells, Source, tallies=tallies, 
                           N_hist=10000)

# Run
simulator.run()
