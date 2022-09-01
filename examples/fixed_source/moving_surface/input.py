import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Three slab layers with different materials

# Set materials
m     = mcdc.material(capture=np.array([0.1]), scatter=np.array([[0.9]]))
m_abs = mcdc.material(capture=np.array([0.9]), scatter=np.array([[0.1]]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
s2 = mcdc.surface('plane-x', x=[1.0, 2.0,  5.0,  1.0],
                             t=[0.0, 12.0, 20.0, 20.0])
s3 = mcdc.surface('plane-x', x=6.0, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m)
mcdc.cell([+s2, -s3], m_abs)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the medium

mcdc.source(x=[0.0, 6.0], time=[0.0, 25.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally(scores=['flux', 'current'],
           x=np.linspace(0.0, 6.0, 61), t=np.linspace(0.0, 25.0, 51))

# Setting
mcdc.setting(N_particle=int(1E7))

# Run
mcdc.run()
