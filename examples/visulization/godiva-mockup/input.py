import numpy as np
import mcdc, h5py

# =============================================================================
# Materials
# =============================================================================

m_abs = mcdc.material(capture=np.array([1e5]), speed=np.array([1e3]), name="water")
m_void = mcdc.material(
    capture=np.array([5e-5]),
    scatter=np.array([[5e-5]]),
    speed=np.array([1e3]),
    name="source",
)

# =============================================================================
# Set surfaces
# =============================================================================

# For cube boundaries
cube_x0 = mcdc.surface("plane-x", x=-22.0, bc="vacuum")
cube_x1 = mcdc.surface("plane-x", x=22.0, bc="vacuum")
cube_y0 = mcdc.surface("plane-y", y=-12.0, bc="vacuum")
cube_y1 = mcdc.surface("plane-y", y=12.0, bc="vacuum")
cube_z0 = mcdc.surface("plane-z", z=-12.0, bc="vacuum")
cube_z1 = mcdc.surface("plane-z", z=12.0, bc="vacuum")

# For the 3-part hollow sphere
sp_left = mcdc.surface("sphere", center=[-2.0, 0.0, 0.0], radius=6.0)
sp_center = mcdc.surface("sphere", center=[0.0, 0.0, 0.0], radius=6.0)
sp_right = mcdc.surface("sphere", center=[2.0, 0.0, 0.0], radius=6.0)
pl_x0 = mcdc.surface("plane-x", x=-3.5)
pl_x1 = mcdc.surface("plane-x", x=-1.5)
pl_x2 = mcdc.surface("plane-x", x=1.5)
pl_x3 = mcdc.surface("plane-x", x=3.5)

# For the moving rod
cy = mcdc.surface("cylinder-x", center=[0.0, 0.0], radius=0.5)
pl_rod0 = mcdc.surface("plane-x", x=[-22.0, 22.0 - 12.0], t=[0.0, 5.0])
pl_rod1 = mcdc.surface("plane-x", x=[-22.0 + 12.0, 22.0], t=[0.0, 5.0])

# =============================================================================
# Set cells
# =============================================================================

# Moving rod
mcdc.cell([-cy, +pl_rod0, -pl_rod1], m_void)

# 3-part hollow shpere
mcdc.cell([-sp_left, -pl_x0, +cy], m_void)
mcdc.cell([-sp_center, +pl_x1, -pl_x2, +cy], m_void)
mcdc.cell([-sp_right, +pl_x3, +cy], m_void)

# Surrounding water
# Left of rod
mcdc.cell([-cy, +cube_x0, -pl_rod0], m_abs)
# Right of rod
mcdc.cell([-cy, +pl_rod1, -cube_x1], m_abs)
# The rest
mcdc.cell(
    [+cy, +sp_left, +cube_x0, -pl_x0, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs
)
mcdc.cell([+cy, +pl_x0, -pl_x1, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs)
mcdc.cell([+sp_center, +pl_x1, -pl_x2, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs)
mcdc.cell([+cy, +pl_x2, -pl_x3, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs)
mcdc.cell(
    [+cy, +sp_right, +pl_x3, -cube_x1, +cube_y0, -cube_y1, +cube_z0, -cube_z1], m_abs
)

# =============================================================================
# Set source
# =============================================================================

mcdc.source(x=[-22.0, 22.0], time=[0.0, 5.0], isotropic=True)

mcdc.visualize(
    start_time=0, end_time=1, tick_interval=0.1, material_colors={"water": [1, 0, 0]}
)
# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: z-integrated flux (X-Y section view)

"""
mcdc.tally(
    scores=["flux"],
    x=np.linspace(-22.0, 22.0, 84+1),
    y=np.linspace(-12.0, 12.0, 24+1),
    t=np.linspace(0.0, 5.0, 50+1),
)

# Setting
mcdc.setting(N_particle=1e6)

# Run
mcdc.run()
"""
