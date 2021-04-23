import numpy as np
import sys, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))

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
time = mcdc.DistDelta(0.0)

# Create the source
Source = mcdc.SourceSimple(pos,dir,g,time,cell=C1)

# =============================================================================
# Set filters and tallies
# =============================================================================

energy_filter = mcdc.FilterEnergyGroup(np.arange(G))

T = mcdc.Tally('tally', scores=['flux'], energy_filter=energy_filter)

tallies = [T]

# =============================================================================
# Set and run simulator (for each value of N)
# =============================================================================

# Set simulator
simulator = mcdc.Simulator(speeds, cells, Source, tallies=tallies)

# Set VRT
simulator.set_vrt(continuous_capture=True,wgt_roulette=0.25)

# Cases to run
N_hist_list = np.logspace(0,7,15).astype(int)

for N_hist in N_hist_list:
    # Set number of histories
    simulator.N_hist = N_hist
    
    # Reset output name
    simulator.output = 'output_N=%i'%N_hist
    
    # Run!
    simulator.run()
