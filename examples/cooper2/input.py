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
Sx1 = mcdc.SurfacePlaneX(0.0, "vacuum")
Sx2 = mcdc.SurfacePlaneX(10.0)
Sx3 = mcdc.SurfacePlaneX(12.0)
Sx4 = mcdc.SurfacePlaneX(20.0, "vacuum")
Sy1 = mcdc.SurfacePlaneY(0.0, "vacuum")
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
# Set and run simulator
# =============================================================================

# Set simulator and tally
simulator = mcdc.Simulator(cells=cells, sources=sources, N_hist=1E8)
simulator.set_tally(scores=['flux'], 
                    x=np.linspace(0.0, 20.0, 41), y=np.linspace(0.0, 20.0, 41))
simulator.implicit_capture = True

# Run
simulator.run()
