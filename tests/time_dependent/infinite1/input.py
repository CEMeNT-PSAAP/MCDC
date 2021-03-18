import sys, os
import numpy as np

# Get path to mcdc
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))),'mcdc'))
import mcdc


# =============================================================================
# Set material XS
# =============================================================================

with np.load('XS.npz') as data:
    speeds = data['v']        # cm/s
    SigmaT = data['SigmaT']   # /cm
    SigmaS = data['SigmaS']
    SigmaF = data['SigmaF']
    nu     = data['nu']
    E      = data['E']        # eV
G     = len(speeds)
E_mid = 0.5*(E[:-1] + E[1:])
dE    = E[1:] - E[:-1]

# Augment with uniform leakage XS
SigmaL  = 0.24 # /cm
SigmaT += SigmaL

# Set material
M1 = mcdc.Material(SigmaT,SigmaS,nu,SigmaF)

# =============================================================================
# Set cells
# =============================================================================

# Set surfaces
S0 = mcdc.SurfacePlaneX(-1E10,"reflective")
S1 = mcdc.SurfacePlaneX(1E10,"reflective")

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
g = mcdc.DistDelta(G-1)

# Time distribution
time= mcdc.DistUniform(0.0,1E-4)

# Create the source
Source = mcdc.SourceSimple(pos,dir,g,time,cell=C1)

# =============================================================================
# Set filters and tallies
# =============================================================================

time_filter = mcdc.FilterTime(np.append(np.array(0.0),np.logspace(-7,-4,161)))
energy_filter = mcdc.FilterEnergyGroup(np.arange(G))

T = mcdc.Tally('tally', scores=['flux', 'flux-edge'],
               energy_filter=energy_filter,
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