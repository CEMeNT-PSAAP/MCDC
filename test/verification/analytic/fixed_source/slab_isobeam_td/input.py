import numpy as np
import sys

import mcdc

N_particle = int(sys.argv[2])

# =============================================================================
# Set model
# =============================================================================
# Finite homogeneous pure-absorbing slab

# Set materials
m = mcdc.material(capture=np.array([1.0]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
s2 = mcdc.surface('plane-x', x=5.0, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================
# Isotropic beam from left-end

mcdc.source(point=[1E-10,0.0,0.0], time=[0.0, 5.0], 
            white_direction=[1.0, 0.0, 0.0])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
mcdc.tally(scores=['flux', 'flux-x', 'flux-t'], x=np.linspace(0.0, 5.0, 51), 
           t=np.linspace(0.0, 5.0, 51))

# Setting
mcdc.setting(N_particle=N_particle, output='output_'+str(N_particle),
             progress_bar=False)

# Run
mcdc.run()
