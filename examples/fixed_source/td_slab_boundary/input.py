import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================

# Set materials
m = mcdc.material(capture=np.array([0.25]), speed=np.array([1.0]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=0.0,  bc="vacuum")
s2 = mcdc.surface('plane-x', x=10.0, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(point=[1E-10,0.0,0.0], time=[0.0,20.0], 
            white_direction=[1.0, 0.0, 0.0])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
mcdc.tally(scores=['flux-t'], x=np.linspace(0.0, 10.0, 201), 
           t=np.linspace(0.0, 20.0, 21))

# Setting
mcdc.setting(N_particle=1E5)

# Run
mcdc.run()
