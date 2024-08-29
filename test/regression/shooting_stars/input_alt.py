import numpy as np

import mcdc


# =============================================================================
# Materials
# =============================================================================

# Set materials
fuel = mcdc.material(
    capture=np.array([0.45]),
    fission=np.array([0.55]),
    nu_p=np.array([2.5]),
)

cover = mcdc.material(
    capture=np.array([0.05]),
    scatter=np.array([[0.95]]),
)

water = mcdc.material(
    capture=np.array([0.02]),
    scatter=np.array([[0.08]]),
)

# =============================================================================
# Set an assembly
# =============================================================================

# Surfaces
cylinder_z1 = mcdc.surface("cylinder-z", center=[-5.0, 0.0], radius=1.0)
cylinder_x = mcdc.surface("cylinder-x", center=[0.0, 0.0], radius=1.0)
cylinder_z2 = mcdc.surface("cylinder-z", center=[5.0, 0.0], radius=1.0)

top_z = mcdc.surface("plane-z", z=2.5)
bot_z = mcdc.surface("plane-z", z=-2.5)
top_x1 = mcdc.surface("plane-x", x=2.5 - 5.0)
bot_x1 = mcdc.surface("plane-x", x=-2.5 - 5.0)
top_x2 = mcdc.surface("plane-x", x=2.5 + 5.0)
bot_x2 = mcdc.surface("plane-x", x=-2.5 + 5.0)

sphere1 = mcdc.surface("sphere", center=[-5.0, 0.0, 0.0], radius=3.0)
sphere2 = mcdc.surface("sphere", center=[5.0, 0.0, 0.0], radius=3.0)

# Cells
pellet_z1 = -cylinder_z1 & +bot_z & -top_z
pellet_z2 = -cylinder_z2 & +bot_z & -top_z
pellet_x1 = -cylinder_x & +bot_x1 & -top_x1
pellet_x2 = -cylinder_x & +bot_x2 & -top_x2
shooting_star1 = pellet_z1 | pellet_x1
shooting_star2 = pellet_z2 | pellet_x2
fuel_shooting_star1 = mcdc.cell(shooting_star1, fuel)
fuel_shooting_star2 = mcdc.cell(shooting_star2, fuel)
cover_sphere1 = mcdc.cell(-sphere1 & ~shooting_star1, cover)
cover_sphere2 = mcdc.cell(-sphere2 & ~shooting_star2, cover)

min_x = mcdc.surface("plane-x", x=-10.0, bc="vacuum")
max_x = mcdc.surface("plane-x", x=10.0, bc="vacuum")
min_y = mcdc.surface("plane-y", y=-5.0, bc="vacuum")
max_y = mcdc.surface("plane-y", y=5.0, bc="vacuum")
min_z = mcdc.surface("plane-z", z=-5.0, bc="vacuum")
max_z = mcdc.surface("plane-z", z=5.0, bc="vacuum")

water_tank = mcdc.cell(
    +sphere1 & +sphere2 & +min_x & -max_x & +min_y & -max_y & +min_z & -max_z, water
)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the domain

mcdc.source()

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average and cell-edge angular fluxes and currents
mcdc.tally.mesh_tally(
    x=np.linspace(-10, 10, 201),
    z=np.linspace(-5, 5, 101),
    scores=["fission"],
)

# Setting
mcdc.setting(N_particle=100000, active_bank_buff=1000, output_name="output_alt")

# Run
mcdc.run()
"""
colors = {
    fuel: 'red',
    cover: 'gray',
    water: 'blue',
}
mcdc.visualize('xz', y=0.0, x=[-11., 11.], z=[-6, 6], pixel=(400, 400), colors=colors)
"""
