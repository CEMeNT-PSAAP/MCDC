import numpy as np
import sys, h5py, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../')

import mcdc

# =============================================================================
# Set material XS
# =============================================================================

#with np.load('test_regression_multigroup_G12_k_XS.npz', allow_pickle=True) as data:
with np.load('test_regression_multigroup_G12_k_XS.npz') as data:
    speeds = data['v']        # cm/s
    SigmaT = data['SigmaT']   # /cm
    SigmaS = data['SigmaS']
    SigmaF = data['SigmaF']
    SigmaC = SigmaT - np.sum(SigmaS,0) - np.sum(SigmaF,0)
    nu     = data['nu']
G = len(speeds)

# Augment with uniform leakage XS
SigmaL  = 0.14 # /cm
SigmaC += SigmaL

# Set material
M = mcdc.Material(SigmaC, SigmaS, SigmaF, nu)

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
                           N_hist=1000)

# Set k-eigenvalue mode parameters
simulator.set_kmode(N_iter=110)

def test_regression_multigroup_G12_k():
    # Run
    simulator.run()

    # Ans
    with h5py.File('output.h5', 'r') as f:
        phi = f['tally/flux/mean'][:]
        k   = f['keff'][:]
        
    # Sol
    with h5py.File('test_regression_multigroup_G12_k_solution_h5', 'r') as f:
        phi_ref = f['tally/flux/mean'][:]
        k_ref   = f['keff'][:]

    assert phi.all() == phi_ref.all()
    assert k.all() == k_ref.all()
