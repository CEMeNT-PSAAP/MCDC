import numpy as np
import sys

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic square source at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark), new computations by William Bennett
# Effective scattering ratio c = 1

# Set materials
m = mcdc.material(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2]),
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================
# Square pulse at center from x = -.5 to .5, t = 0

mcdc.source(x=[-0.5, 0.5], isotropic=True, time=[1e-10, 1e-10])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average, cell-edge, and time-edge scalar fluxes
mcdc.tally(
    scores=["flux"],
    x=np.linspace(-20.5, 20.5, 202),
    t=np.linspace(0.0, 20.0, 21),
)

# Setting
mcdc.setting(N_particle=1000)

# Run
mcdc.run()
