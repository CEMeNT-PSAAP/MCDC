import numpy as np
import sys, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set material XS
# =============================================================================

sys.path.append('../')
with np.load('SHEM-361.npz') as data:
    speeds = data['v']        # cm/s
    SigmaC = data['SigmaC']   # /cm
    SigmaS = data['SigmaS']
    SigmaF = data['SigmaF']
    nu_p   = data['nu_p']
    nu_d   = data['nu_d']
    chi_d  = data['chi_d']
    chi_p  = data['chi_p']
    decay  = data['lamd']
    G = data['G']

# Set material
M = mcdc.Material(SigmaC, SigmaS, SigmaF, nu_p, nu_d, chi_p, chi_d)

# =============================================================================
# Set cells
# =============================================================================

# Set surfaces
S0 = mcdc.SurfacePlaneX(-1E10,"reflective")
S1 = mcdc.SurfacePlaneX(1E10,"reflective")

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
g = mcdc.DistDelta(G-1)

# Time distribution
time = mcdc.DistDelta(0.0)

# Create the source
Src = mcdc.SourceSimple(pos,dir,g,time,cell=C)
sources = [Src]

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
simulator = mcdc.Simulator(speeds, cells, sources, tallies=tallies, 
                           N_hist=1000, decay=decay)

# Set k-eigenvalue mode parameters
simulator.set_kmode(N_iter=110, alpha_mode=True)

# Run
simulator.run()
