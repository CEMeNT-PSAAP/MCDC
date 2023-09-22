import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Three slab layers with different purely-absorbing materials

# Set materials
m1 = mcdc.material(capture=np.array([1.0]))
m2 = mcdc.material(capture=np.array([0]))


# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=2.0)
s3 = mcdc.surface("plane-x", x=4.0)
s4 = mcdc.surface("plane-x", x=6.0, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m1)
mcdc.cell([+s2, -s3], m2)
mcdc.cell([+s3, -s4], m1)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the domain

mcdc.source(x=[0.0, 2.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average and cell-edge angular fluxes and currents
mcdc.tally(
    scores=["flux", "current"],
    x=np.linspace(0.0, 6.0, 61)
)

# Setting
mcdc.setting(N_particle=1e4,active_bank_buff=1000000,save_input_deck=True)
mcdc.domain_decomp(x=np.linspace(0.0,6.0,7),work_ratio=([3,1,1,1,3,1]))

# Run
mcdc.run()
