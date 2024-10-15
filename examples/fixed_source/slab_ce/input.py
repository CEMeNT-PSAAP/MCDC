import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================

rho_uo2 = 10.97
rho_h2o = 0.997
rho_b4c = 2.52

a_u235 = 0.05
a_u238 = 1.0 - a_u235

A_uo2 = 270.03
A_h2o = 18.01528
A_b4c = 55.255

N_avo = 6.023e23

N_uo2 = rho_uo2 * N_avo / A_uo2 * 1e-24
N_h2o = rho_h2o * N_avo / A_h2o * 1e-24
N_b4c = rho_b4c * N_avo / A_b4c * 1e-24

N_u235 = a_u235 * N_uo2
N_u238 = a_u238 * N_uo2
N_o16_uo2 = 2.0 * N_uo2

N_h1 = 2.0 * N_h2o
N_o16_h2o = N_h2o

N_b10 = 4.0 * N_b4c
N_c12 = N_b4c

# Set materials
uo2 = mcdc.material(
    [
        ["U235", N_u235],
        ["U238", N_u238],
        ["O16", N_o16_uo2],
    ]
)
h2o = mcdc.material(
    [
        ["H1", N_h1],
        ["O16", N_o16_h2o],
    ]
)
b4c = mcdc.material(
    [
        ["B10", N_b10],
        ["C12", N_c12],
    ]
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
s2 = mcdc.surface("plane-x", x=0.5)
s3 = mcdc.surface("plane-x", x=1.5)
s4 = mcdc.surface("plane-x", x=2.0, bc="reflective")

# Set cells
mcdc.cell(+s1 & -s2, uo2)
mcdc.cell(+s2 & -s3, h2o)
mcdc.cell(+s3 & -s4, b4c)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(
    x=[0.95, 1.05], energy=np.array([[14e6 - 1, 14e6 + 1], [1.0, 1.0]]), isotropic=True
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally.mesh_tally(
    scores=["flux"], x=np.linspace(0.0, 2.0, 21), E=np.array([0.0, 1.0, 20e6])
)
mcdc.setting(N_particle=1e3)
mcdc.run()
