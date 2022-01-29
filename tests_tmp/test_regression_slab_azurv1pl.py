import numpy as np
import sys, h5py, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../')

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
Src = mcdc.SourceSimple(pos,dir,g,time,cell=C)
sources = [Src]

# =============================================================================
# Set filters and tallies
# =============================================================================

# Load grids
grid = np.load('test_regression_slab_azurv1pl.npz')
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
simulator = mcdc.Simulator(speeds, cells, sources, tallies=tallies, 
                           N_hist=10000)

# Set population control and census
simulator.set_pct(census_time=np.array([20.0]))

def test_regression_slab_azurv1pl():
    # Run
    simulator.run()

    # Ans
    with h5py.File('output.h5', 'r') as f:
        phi         = f['tally/flux/mean'][:]
        phi_sd      = f['tally/flux/sdev'][:]
        phi_edge    = f['tally/flux-edge/mean'][:]
        phi_edge_sd = f['tally/flux-edge/sdev'][:]
        phi_face    = f['tally/flux-face/mean'][:]
        phi_face_sd = f['tally/flux-face/sdev'][:]
    os.remove('output.h5')

    # Sol
    with h5py.File('test_regression_slab_azurv1pl_solution_h5', 'r') as f:
        phi_ref         = f['tally/flux/mean'][:]
        phi_sd_ref      = f['tally/flux/sdev'][:]
        phi_edge_ref    = f['tally/flux-edge/mean'][:]
        phi_edge_sd_ref = f['tally/flux-edge/sdev'][:]
        phi_face_ref    = f['tally/flux-face/mean'][:]
        phi_face_sd_ref = f['tally/flux-face/sdev'][:]

    assert phi.all() == phi_ref.all()
    assert phi_sd.all() == phi_sd_ref.all()
    assert phi_edge.all() == phi_edge_ref.all()
    assert phi_edge_sd.all() == phi_edge_sd_ref.all()
    assert phi_face.all() == phi_face_ref.all()
    assert phi_face_sd.all() == phi_face_sd_ref.all()
