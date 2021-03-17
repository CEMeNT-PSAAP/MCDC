import sys, os
import numpy as np

# Get path to mcdc
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))),'mcdc'))
import mcdc


# =============================================================================
# Set material XS
# =============================================================================

speeds = [1.0]

SigmaT = np.array([1.0])
nu     = np.array([0.0])
SigmaF = np.array([[0.0]])
SigmaS = np.array([[[0.9]]])
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