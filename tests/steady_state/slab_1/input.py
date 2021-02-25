import sys
sys.path.append('C:\\Users\\Ilham\\iCloudDrive\\Work\\MCDC\\mcdc')
import numpy as np

import mcdc


# =============================================================================
# Set material XS
# =============================================================================

speeds = [1.0]

SigmaT = np.array([1.0])
SigmaA = np.array([1.0])
nu     = np.array([0.0])
SigmaF = np.array([[0.0]])
SigmaS = np.array([[[0.0]]])
M1 = mcdc.Material(SigmaT,SigmaS,nu,SigmaF)

SigmaT = np.array([1.5])
SigmaA = np.array([1.5])
M2 = mcdc.Material(SigmaT,SigmaS,nu,SigmaF)

SigmaT = np.array([2.0])
SigmaA = np.array([2.0])
M3 = mcdc.Material(SigmaT,SigmaS,nu,SigmaF)

# =============================================================================
# Set cells
# =============================================================================

# Set surfaces
S0 = mcdc.SurfacePlaneX(0.0,"vacuum")
S1 = mcdc.SurfacePlaneX(2.0,"transmission")
S2 = mcdc.SurfacePlaneX(4.0,"transmission")
S3 = mcdc.SurfacePlaneX(6.0,"vacuum")

# Set cells
C1 = mcdc.Cell([[S0,+1],[S1,-1]],M2)
C2 = mcdc.Cell([[S1,+1],[S2,-1]],M3)
C3 = mcdc.Cell([[S2,+1],[S3,-1]],M1)
cells = [C1, C2, C3]

# =============================================================================
# Set source
# =============================================================================

# Position distribution
pos = mcdc.DistPoint(mcdc.DistDelta(mcdc.small_kick), mcdc.DistDelta(0.0), 
                             mcdc.DistDelta(0.0))
# Direction distribution
dir = mcdc.DistPoint(mcdc.DistDelta(1.0), mcdc.DistDelta(0.0), 
                             mcdc.DistDelta(0.0))
# Energy group distribution
g = mcdc.DistDelta(0)

# Time distribution
time = mcdc.DistDelta(0.0)

# Create the source
Source = mcdc.SourceSimple(pos,dir,g,time,cell=C1)

# =============================================================================
# Set filters and tallies
# =============================================================================

spatial_filter = mcdc.FilterPlaneX(np.linspace(0.0, 6.0, 61))

T = mcdc.Tally('tally', scores=['flux', 'flux-face'], 
               spatial_filter=spatial_filter)

tallies = [T]

# =============================================================================
# Set and run simulator (for each value of N)
# =============================================================================

# Set simulator
simulator = mcdc.Simulator(speeds, cells, Source, tallies=tallies)

# Cases to run
N_hist_list = np.logspace(0,5,11).astype(int)

for N_hist in N_hist_list:
    # Set number of histories
    simulator.N_hist = N_hist
    
    # Reset output name
    simulator.output = 'output_N=%i'%N_hist
    
    # Run!
    simulator.run()