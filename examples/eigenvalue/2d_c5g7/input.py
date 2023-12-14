import h5py
import numpy as np

import mcdc

# =============================================================================
# Materials
# =============================================================================

# Load material data
lib = h5py.File("2d_c5g7_xs.h5", "r")


# Materials
def set_mat(mat):
    return mcdc.material(
        capture=mat["capture"][:],
        scatter=mat["scatter"][:],
        fission=mat["fission"][:],
        nu_p=mat["nu"][:],
        chi_p=mat["chi"][:],
    )


mat_uo2 = set_mat(lib["uo2"])
mat_mox43 = set_mat(lib["mox43"])
mat_mox7 = set_mat(lib["mox70"])
mat_mox87 = set_mat(lib["mox87"])
mat_gt = set_mat(lib["gt"])
mat_fc = set_mat(lib["fc"])
mat_mod = set_mat(lib["mod"])

# =============================================================================
# Pin cells
# =============================================================================

pitch = 1.26
radius = 0.54

# Surfaces
cy = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=radius)

# Cells
uo2 = mcdc.cell([-cy], mat_uo2)
mox4 = mcdc.cell([-cy], mat_mox43)
mox7 = mcdc.cell([-cy], mat_mox7)
mox8 = mcdc.cell([-cy], mat_mox87)
gt = mcdc.cell([-cy], mat_gt)
fc = mcdc.cell([-cy], mat_fc)
mod = mcdc.cell([+cy], mat_mod)
modi = mcdc.cell([-cy], mat_mod)  # For all-water lattice

# Universes
u = mcdc.universe([uo2, mod])["ID"]
l = mcdc.universe([mox4, mod])["ID"]
m = mcdc.universe([mox7, mod])["ID"]
n = mcdc.universe([mox8, mod])["ID"]
g = mcdc.universe([gt, mod])["ID"]
f = mcdc.universe([fc, mod])["ID"]
w = mcdc.universe([modi, mod])["ID"]

# =============================================================================
# Assemblies
# =============================================================================

# Lattices
lattice_uo2 = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, g, u, u, g, u, u, g, u, u, u, u, u],
        [u, u, u, g, u, u, u, u, u, u, u, u, u, g, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, g, u, u, g, u, u, g, u, u, g, u, u, g, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, g, u, u, g, u, u, f, u, u, g, u, u, g, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, g, u, u, g, u, u, g, u, u, g, u, u, g, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, g, u, u, u, u, u, u, u, u, u, g, u, u, u],
        [u, u, u, u, u, g, u, u, g, u, u, g, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
    ],
)

lattice_mox = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, m, m, m, m, g, m, m, g, m, m, g, m, m, m, m, l],
        [l, m, m, g, m, n, n, n, n, n, n, n, m, g, m, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, g, n, n, g, n, n, g, n, n, g, n, n, g, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, g, n, n, g, n, n, f, n, n, g, n, n, g, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, g, n, n, g, n, n, g, n, n, g, n, n, g, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, m, g, m, n, n, n, n, n, n, n, m, g, m, m, l],
        [l, m, m, m, m, g, m, m, g, m, m, g, m, m, m, m, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
    ],
)

lattice_mod = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch * 17, 1],
    y=[-pitch * 17 / 2, pitch * 17, 1],
    universes=[[w]],
)

# Assembly cells
# Surfaces
x0 = mcdc.surface("plane-x", x=-pitch * 17 / 2)
x1 = mcdc.surface("plane-x", x=pitch * 17 / 2)
y0 = mcdc.surface("plane-y", y=-pitch * 17 / 2)
y1 = mcdc.surface("plane-y", y=pitch * 17 / 2)
# Cells
assembly_uo2 = mcdc.cell([+x0, -x1, +y0, -y1], lattice_uo2)
assembly_mox = mcdc.cell([+x0, -x1, +y0, -y1], lattice_mox)
assembly_mod = mcdc.cell([+x0, -x1, +y0, -y1], lattice_mod)

# Set assemblies in their respective universes
u_ = mcdc.universe([assembly_uo2])["ID"]
m_ = mcdc.universe([assembly_mox])["ID"]
w_ = mcdc.universe([assembly_mod])["ID"]

# =============================================================================
# Root universe: core
# =============================================================================

# Lattice
lattice_core = mcdc.lattice(
    x=[-pitch * 17 * 3 / 2, pitch * 17, 3],
    y=[-pitch * 17 * 3 / 2, pitch * 17, 3],
    universes=[[u_, m_, w_], [m_, u_, w_], [w_, w_, w_]],
)

# Core cell
# Surfaces
x0_ = mcdc.surface("plane-x", x=0.0, bc="reflective")
x1_ = mcdc.surface("plane-x", x=pitch * 17 * 3, bc="vacuum")
y0_ = mcdc.surface("plane-y", y=-pitch * 17 * 3, bc="vacuum")
y1_ = mcdc.surface("plane-y", y=0.0, bc="reflective")
# Cell
core = mcdc.cell(
    [+x0_, -x1_, +y0_, -y1_],
    lattice_core,
    lattice_center=[pitch * 17 * 3 / 2, -pitch * 17 * 3 / 2, 0.0],
)

# Root universe
mcdc.universe([core], root=True)

# =============================================================================
# Set source
# =============================================================================
# Uniform in energy

source = mcdc.source(
    x=[0.0, pitch * 17 * 2], y=[-pitch * 17 * 2, 0.0], energy=np.ones(7)
)

# =============================================================================
# Set tally and parameter, and then run mcdc
# =============================================================================

# Tally
x_grid = np.linspace(0.0, pitch * 17 * 3, 17 * 3 + 1)
y_grid = np.linspace(-pitch * 17 * 3, 0.0, 17 * 3 + 1)
mcdc.tally(scores=["flux"], x=x_grid, y=y_grid)

# Setting
mcdc.setting(N_particle=1e3)
mcdc.eigenmode(N_inactive=10, N_active=30, gyration_radius="infinite-z")
mcdc.population_control()

# Run
mcdc.run()
