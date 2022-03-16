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
m = mcdc.material(capture=np.array([0.2]), scatter=np.array([[0.8]]))
m_barrier = mcdc.material(capture=np.array([1.0]), scatter=np.array([[4.0]]))

# Set surfaces
sx1 = mcdc.surface('plane-x', x=0.0,  bc="reflective")
sx2 = mcdc.surface('plane-x', x=10.0)
sx3 = mcdc.surface('plane-x', x=12.0)
sx4 = mcdc.surface('plane-x', x=20.0, bc="vacuum")
sy1 = mcdc.surface('plane-y', y=0.0,  bc="vacuum")
sy2 = mcdc.surface('plane-y', y=10.0)
sy3 = mcdc.surface('plane-y', y=20.0, bc="reflective")

# Set cells
c1 = mcdc.cell([+sx1, -sx2, +sy1, -sy2], m)
c2 = mcdc.cell([+sx3, -sx4, +sy1, -sy2], m)
c3 = mcdc.cell([+sx1, -sx4, +sy2, -sy3], m)
c_barrier = mcdc.cell([+sx2, -sx3, +sy1, -sy2], m_barrier)
cells = [c1, c2, c3, c_barrier]

# =============================================================================
# Set source
# =============================================================================

source = mcdc.source(x=[0.0, 5.0], y=[0.0, 5.0], isotropic=True)

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

mcdc.set_problem(cells, source, N_hist=1E3)
mcdc.set_tally(scores=['flux'], 
                    x=np.linspace(0.0, 20.0, 41), y=np.linspace(0.0, 20.0, 41))
mcdc.set_implicit_capture(True)
mcdc.run()
