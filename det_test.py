import mcdc
import numpy as np


# =============================================================================
# Set model
# =============================================================================

# Set materials
m = mcdc.material(capture = np.array([.1]),
                  scatter = np.array([[.9]]),
                  fission = np.array([0.0]),
                  nu_p    = np.array([0.0]),
                  speed   = np.array([1.0]))

# Set surfaces
s1 = mcdc.surface('plane-x', x=-1E10, bc="vacuum")
s2 = mcdc.surface('plane-x', x=1E10,  bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(point=[0.0,0.0,0.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally
mcdc.tally(scores=['flux'],
           x=np.linspace(-5.25, 5.25, 2000),
           t=np.linspace(0.0, 1, 10))

# Setting
mcdc.setting(N_particle=1E6)

# Run
mcdc.run()