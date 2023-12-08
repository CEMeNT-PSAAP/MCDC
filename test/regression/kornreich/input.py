import numpy as np
import h5py

import mcdc


# =========================================================================
# Set model
# =========================================================================
# Based on Kornreich, ANE 2004, 31, 1477-1494,
# DOI: 10.1016/j.anucene.2004.03.012

# Set materials
m1 = mcdc.material(
    capture=np.array([0.0]),
    scatter=np.array([[0.9]]),
    fission=np.array([0.1]),
    nu_p=np.array([6.0]),
)
m2 = mcdc.material(
    capture=np.array([0.68]),
    scatter=np.array([[0.2]]),
    fission=np.array([0.12]),
    nu_p=np.array([2.5]),
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="vacuum")
s2 = mcdc.surface("plane-x", x=1.5)
s3 = mcdc.surface("plane-x", x=2.5, bc="vacuum")

# Set cells
mcdc.cell([+s1, -s2], m1)
mcdc.cell([+s2, -s3], m2)

# =========================================================================
# Set source
# =========================================================================

mcdc.source(x=[0.0, 2.5], isotropic=True)

# =========================================================================
# Set tally, setting, and run mcdc
# =========================================================================

# Tally
x = np.array(
    [
        0.0,
        0.15,
        0.3,
        0.45,
        0.6,
        0.75,
        0.9,
        1.05,
        1.2,
        1.35,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2,
        2.1,
        2.2,
        2.3,
        2.4,
        2.5,
    ]
)
scores = ["flux"]
mcdc.tally(scores=scores, x=x)

# Setting
mcdc.setting(N_particle=100, progress_bar=False, census_bank_buff=2.0)
mcdc.eigenmode(N_inactive=1, N_active=2, gyration_radius="only-x")

# Run
mcdc.run()
