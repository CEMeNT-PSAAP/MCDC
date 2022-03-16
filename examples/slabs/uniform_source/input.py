import numpy as np
import sys

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set cells
# =============================================================================
# Three slab layers with different materials

# Set materials
m1 = mcdc.material(capture=np.array([1.0]))
m2 = mcdc.material(capture=np.array([1.5]))
m3 = mcdc.material(capture=np.array([2.0]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
s2 = mcdc.surface('plane-x', x=2.0)
s3 = mcdc.surface('plane-x', x=4.0)
s4 = mcdc.surface('plane-x', x=6.0, bc="vacuum")

# Set cells
c1 = mcdc.cell([+s1, -s2], m2)
c2 = mcdc.cell([+s2, -s3], m3)
c3 = mcdc.cell([+s3, -s4], m1)

cells = [c1, c2, c3]

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the medium

source = mcdc.source(x=[0.0, 6.0], isotropic=True)

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

mcdc.set_problem(cells, source, N_hist=1E4)
mcdc.set_tally(scores=['flux', 'current'], x=np.linspace(0.0, 6.0, 61))
mcdc.run()
