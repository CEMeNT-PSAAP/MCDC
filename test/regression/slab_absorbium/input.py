import numpy as np
import sys

import mcdc


# =============================================================================
# Set model
# =============================================================================
# Three slab layers with different purely-absorbing materials

# Set materials
m1 = mcdc.material(capture=np.array([1.0]))
m2 = mcdc.material(capture=np.array([1.5]))
m3 = mcdc.material(capture=np.array([2.0]))

# Set surfaces
s1 = mcdc.surface("plane-z", z=0.0, bc="vacuum")
s2 = mcdc.surface("plane-z", z=2.0)
s3 = mcdc.surface("plane-z", z=4.0)
s4 = mcdc.surface("plane-z", z=6.0, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m2)
mcdc.cell([+s2, -s3], m3)
mcdc.cell([+s3, -s4], m1)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the domain

mcdc.source(z=[0.0, 6.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average and cell-edge angular fluxes and currents
mcdc.tally(
    scores=["flux", "current"],
    z=np.linspace(0.0, 6.0, 61),
    mu=np.linspace(-1.0, 1.0, 32 + 1),
)

# Setting
mcdc.setting(N_particle=100)

# Run
mcdc.run()
