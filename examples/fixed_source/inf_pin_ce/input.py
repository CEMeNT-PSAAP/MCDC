import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================

# Set materials

fuel = mcdc.material(
    [
        ["U235", 0.0005581658948833916],
        ["U238", 0.022404594715383263],
        ["O16", 0.045831301393656466],
        ["O17", 1.7411492132576054e-05],
        ["O18", 9.18996012190109e-05],
    ]
)

water = mcdc.material(
    [
        ["B10", 0.0001357003217727274],
        ["B11", 0.0005489632593207509],
        ["H1", 0.0684556951587359],
        ["H2", 1.0662950611949833e-5],
        ["O16", 0.032785655643293984],
        ["O17", 1.245539986725256e-5],
        ["O18", 6.574084932573092e-5],
    ]
)

# Set surfaces
cy = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=0.45720)
pitch = 1.25984
x1 = mcdc.surface("plane-x", x=-pitch / 2, bc="reflective")
x2 = mcdc.surface("plane-x", x=pitch / 2, bc="reflective")
y1 = mcdc.surface("plane-y", y=-pitch / 2, bc="reflective")
y2 = mcdc.surface("plane-y", y=pitch / 2, bc="reflective")

# Set cells
mcdc.cell(-cy & +x1 & -x2 & +y1 & -y2, fuel)
mcdc.cell(+cy & +x1 & -x2 & +y1 & -y2, water)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(
    x=[-pitch / 2, pitch / 2],
    y=[-pitch / 2, pitch / 2],
    energy=np.array([[1e6 - 1, 1e6 + 1], [1.0, 1.0]]),
    isotropic=True,
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.tally.mesh_tally(
    scores=["flux", "density"],
    E=np.loadtxt("energy_grid.txt"),
    t=np.insert(np.logspace(-8, 2, 50), 0, 0.0),
)
mcdc.setting(N_particle=1e2, active_bank_buff=1000)
mcdc.run()
