import numpy as np
import sys

import mcdc

N_history = int(sys.argv[2])
tag       = sys.argv[3]

# =============================================================================
# Set model
# =============================================================================

# Set materials
m1 = mcdc.material(capture=np.array([0.0]), scatter=np.array([[0.9]]),
                   fission=np.array([0.1]), nu_p=np.array([6.0]))
m2 = mcdc.material(capture=np.array([0.68]), scatter=np.array([[0.2]]),
                   fission=np.array([0.12]), nu_p=np.array([2.5]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=0.0, bc="vacuum")
s2 = mcdc.surface('plane-x', x=1.5)
s3 = mcdc.surface('plane-x', x=2.5, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m1)
mcdc.cell([+s2, -s3], m2)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(x=[0.0, 2.5], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
x  = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.6, 
              1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5])
mcdc.tally(scores=['flux-x'], x=x)

# Setting
mcdc.setting(N_particle=1E5, output='output_'+tag+'_'+str(N_history), 
             progress_bar=False)
mcdc.eigenmode(N_inactive=50, N_active=N_history, gyration_radius='only-x')
mcdc.population_control()

# Run
mcdc.run()
