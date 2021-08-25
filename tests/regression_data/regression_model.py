import numpy as np
import sys

from context import mcdc

# =============================================================================
# Set material XS
# =============================================================================

speeds = [1.0]

SigmaT = np.array([1.0])
SigmaA = np.array([1.0])
nu     = np.array([0.0])
SigmaF = np.array([[0.0]])
SigmaS = np.array([[0.0]])
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
pos = mcdc.DistPoint(mcdc.DistDelta(mcdc.SMALL), mcdc.DistDelta(0.0), 
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
# Set and run simulator
# =============================================================================

# Set simulator
ss_slab1 = mcdc.Simulator(speeds, cells, Source, tallies=tallies, N_hist=1000)

# =============================================================================
# Set source
# =============================================================================

# Position distribution
pos = mcdc.DistPoint(mcdc.DistUniform(0.0,6.0), mcdc.DistDelta(0.0), 
                             mcdc.DistDelta(0.0))
# Direction distribution
dir = mcdc.DistPointIsotropic()

# Energy group distribution
g = mcdc.DistDelta(0)

# Time distribution
time= mcdc.DistUniform(0.0,6.0)

# Create the source
Source = mcdc.SourceSimple(pos,dir,g,time)

# =============================================================================
# Set filters and tallies
# =============================================================================

spatial_filter = mcdc.FilterPlaneX(np.linspace(0.0, 6.0, 61))

T = mcdc.Tally('tally', scores=['flux', 'flux-face', 'current', 'current-face'], 
               spatial_filter=spatial_filter)

tallies = [T]

# =============================================================================
# Set and run simulator (for each value of N)
# =============================================================================

# Set simulator
ss_slab2 = mcdc.Simulator(speeds, cells, Source, tallies=tallies, N_hist=1000)

# =============================================================================
# Set material XS
# =============================================================================

with np.load('regression_data/XS_inf1.npz') as data:
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
ss_inf1 = mcdc.Simulator(speeds, cells, Source, tallies=tallies, N_hist=1000)

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

# Set simulator
td_inf1 = mcdc.Simulator(speeds, cells, Source, tallies=tallies, N_hist=1000)
