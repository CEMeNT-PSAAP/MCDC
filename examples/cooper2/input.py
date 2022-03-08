import numpy as np
import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../')

import mcdc

# =============================================================================
# Set cells
# =============================================================================
# Three slab layers with different materials

# Set materials
M = mcdc.Material(capture=np.array([0.2]), scatter=np.array([[0.8]]))
M_barrier = mcdc.Material(capture=np.array([1.0]), scatter=np.array([[4.0]]))

# Set surfaces
Sx1 = mcdc.SurfacePlaneX(0.0,  "reflective")
Sx2 = mcdc.SurfacePlaneX(10.0)
Sx3 = mcdc.SurfacePlaneX(12.0)
Sx4 = mcdc.SurfacePlaneX(20.0, "vacuum")
Sy1 = mcdc.SurfacePlaneY(0.0,  "reflective")
Sy2 = mcdc.SurfacePlaneY(10.0)
Sy3 = mcdc.SurfacePlaneY(20.0, "vacuum")

# Set cells
C1 = mcdc.Cell([+Sx1, -Sx2, +Sy1, -Sy2], M)
C2 = mcdc.Cell([+Sx3, -Sx4, +Sy1, -Sy2], M)
C3 = mcdc.Cell([+Sx1, -Sx4, +Sy2, -Sy3], M)
C_barrier = mcdc.Cell([+Sx2, -Sx3, +Sy1, -Sy2], M_barrier)
cells = [C1, C2, C3, C_barrier]

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the medium

position  = mcdc.DistPoint(x=mcdc.DistUniform(0.0,5.0), 
                           y=mcdc.DistUniform(0.0,5.0))
direction = mcdc.DistPointIsotropic()

Src = mcdc.SourceSimple(position=position, direction=direction)
sources = [Src]

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

mcdc.set_problem(cells, sources, N_hist=1E3)
mcdc.set_tally(scores=['flux'], x=np.linspace(0.0, 6.0, 61))
mcdc.set_tally(scores=['flux'], 
                    x=np.linspace(0.0, 20.0, 41), y=np.linspace(0.0, 20.0, 41))
mcdc.set_implicit_capture(True)
mcdc.run()
