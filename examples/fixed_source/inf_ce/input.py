import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================

rho_UO2 = 10.97
rho_H2O = 1.0
N_avo = 6.02E23

M_UO2 = 270.0
M_H2O = 18.0

N_UO2 = rho_UO2*N_avo/M_UO2*1E-24
N_H2O = rho_UO2*N_avo/M_UO2*1E-24

N_U = N_UO2
N_H = 2*N_H2O

N_H1 = N_H
N_U235 = 0.05*N_U
N_U238 = 0.95*N_U

pitch = 1.2
R = 0.45
Vf = np.pi*R**2
Vtot = pitch**2
Vw = Vtot-Vf

N_H1 *= Vw/Vtot
N_U235 *= Vf/Vtot
N_U238 *= Vf/Vtot
N_O16 = 2*N_UO2*Vf/Vtot + N_H2O*Vw/Vtot

N_B10 = 0.01*N_H1/2

# Set materials
mat = mcdc.material(
    [
        ["H1", N_H1],
        ["O16", N_O16],
        ["U235", N_U235],
        ["U238", N_U238],
        ["B10", N_B10],
    ]
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=-1E10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1E10, bc="reflective")

# Set cells
mcdc.cell([+s1, -s2], mat)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(
    point=[0.0,0.0,0.0], energy=np.array([[14e6 - 1, 14e6 + 1], [1.0, 1.0]]), isotropic=True
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

with np.load("SHEM-361.npz") as data:
    E = data["E"]

mcdc.tally(scores=["flux"], E=E, t=np.insert(np.logspace(-8, 2, 50), 0, 0.0))
mcdc.setting(N_particle=1e5, active_bank_buff=10000)
mcdc.run()
