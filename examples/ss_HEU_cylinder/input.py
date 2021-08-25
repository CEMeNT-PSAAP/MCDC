import numpy as np
import sys, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../')

import mcdc


# =============================================================================
# Set material XS
# =============================================================================

# XCOM data
E         = np.array([1.440E-01, 1.860E-01, 7.660E-01, 1.001E+00])*1E6
SigmaT_Al = np.array([1.404E-01, 1.259E-01, 6.981E-02, 6.143E-02])*2.7
SigmaT_Ni = np.array([2.339E-01, 1.702E-01, 7.041E-02, 6.156E-02])*8.9
SigmaT_U  = np.array([2.861E+00, 1.542E+00, 1.072E-01, 7.887E-02])*18.75

G      = len(E)
SigmaS = np.zeros([G,G])
SigmaF = np.zeros([G,G])
nu     = np.zeros(G)
speeds = np.ones(G)

# Set the material objects
M_Al = mcdc.Material(SigmaT_Al,SigmaS,nu,SigmaF)
M_Ni = mcdc.Material(SigmaT_Ni,SigmaS,nu,SigmaF)
M_U  = mcdc.Material(SigmaT_U,SigmaS,nu,SigmaF)

# Void
M_void = mcdc.MaterialVoid(G)


# =============================================================================
# Set cells
# =============================================================================

# Surfaces
HEU_bottom  = 2.0
HEU_top     = 6.0
HEU_radius  = 4.0
S_cy1 = mcdc.SurfaceCylinderZ(0.0,0.0,HEU_radius,name='cy1')
S_cy2 = mcdc.SurfaceCylinderZ(0.0,0.0,5.0,name='cy2')
S_cy3 = mcdc.SurfaceCylinderZ(0.0,0.0,6.0,'vacuum',1,name='cy3')
S_pl1  = mcdc.SurfacePlaneZ(0.0,'vacuum',name='pl1')
S_pl2  = mcdc.SurfacePlaneZ(HEU_bottom,name='pl2')
S_pl3  = mcdc.SurfacePlaneZ(HEU_top,name='pl3')
S_pl4  = mcdc.SurfacePlaneZ(7.0,'vacuum',2,name='pl4')

# Cells
C_HEU       = mcdc.Cell([[S_cy1,-1], [S_pl2,+1], [S_pl3,-1]], M_U, name="HEU")
C_Ni_bottom = mcdc.Cell([[S_cy2,-1], [S_pl1,+1], [S_pl2,-1]], M_Ni, name="Ni_bottom")
C_Ni_top    = mcdc.Cell([[S_cy2,-1], [S_pl3,+1], [S_pl4,-1]], M_Ni, name="Ni_top")
C_Al_inner  = mcdc.Cell([[S_cy1,+1], [S_cy2,-1], [S_pl2,+1], [S_pl3,-1]], M_Al, name="Al_in")
C_Al_outer  = mcdc.Cell([[S_cy2,+1], [S_cy3,-1], [S_pl1,+1], [S_pl4,-1]], M_Al, name="Al_out")

cells = [C_HEU, C_Ni_bottom, C_Ni_top, C_Al_inner, C_Al_outer]


# =============================================================================
# Set source
# =============================================================================

# Position distribution
pos = mcdc.DistPointCylinderZ(0.0, 0.0, HEU_radius, HEU_bottom, HEU_top)

# Direction distribution
dir = mcdc.DistPointIsotropic()

# Energy group distribution
spectrum = np.array([3.3E+04, 3.2E+05, 4.0E+02, 1.5E+03])
g = mcdc.DistGroup(spectrum)

# Time distribution
time = mcdc.DistDelta(0.0)

# Create the source
Source = mcdc.SourceSimple(pos,dir,g,time)

# =============================================================================
# Set filters and tallies
# =============================================================================

spatial_filter = mcdc.FilterSurface([1,2])
energy_filter  = mcdc.FilterEnergyGroup(np.arange(G))

T = mcdc.Tally('tally', scores=['total_crossing'], spatial_filter=spatial_filter, 
               energy_filter=energy_filter)

tallies = [T]

# =============================================================================
# Set and run simulator (for each value of N)
# =============================================================================

# Set simulator
simulator = mcdc.Simulator(speeds, cells, Source, tallies=tallies,
                           N_hist=10000000)

# Turn on some variance reduction techniques (VRT)
#simulator.set_vrt(continuous_capture=True)

# Run
simulator.run()
