import numpy as np
import sys

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
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
# Isotropic pulse at x=t=0
x = np.linspace(-15, 15, int(1e4))
gaussian = np.exp(-4 * x**2)
dx = x[2] - x[1]
edges_x = np.append(x - dx / 2, x[-1] + dx / 2)
for ii, (x1, x2) in enumerate(zip(edges_x[:-1], edges_x[1:])):
    mcdc.source(
        x=[x1, x2],
        prob=gaussian[ii]
        * 0.8862269254527580136490837416705725913987747280611935641069038949,
        time=[1e-10, 1e-10],
    )


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
mcdc.setting(N_particle=100000)

# Run
mcdc.run()
