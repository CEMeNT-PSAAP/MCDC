import h5py
import numpy as np

import mcdc

# =============================================================================
# Materials
# =============================================================================

# Load material data
lib = h5py.File("../../MGXS-C5G7.h5", "r")


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
uo2 = mcdc.cell(-cy, mat_uo2)
mox4 = mcdc.cell(-cy, mat_mox43)
mox7 = mcdc.cell(-cy, mat_mox7)
mox8 = mcdc.cell(-cy, mat_mox87)
gt = mcdc.cell(-cy, mat_gt)
fc = mcdc.cell(-cy, mat_fc)
mod = mcdc.cell(+cy, mat_mod)
modi = mcdc.cell(-cy, mat_mod)  # For all-water lattice

# Universes
u = mcdc.universe([uo2, mod])
l = mcdc.universe([mox4, mod])
m = mcdc.universe([mox7, mod])
n = mcdc.universe([mox8, mod])
g = mcdc.universe([gt, mod])
f = mcdc.universe([fc, mod])
w = mcdc.universe([modi, mod])

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
assembly_uo2 = mcdc.cell(+x0 & -x1 & +y0 & -y1, lattice_uo2)
assembly_mox = mcdc.cell(+x0 & -x1 & +y0 & -y1, lattice_mox)
assembly_mod = mcdc.cell(+x0 & -x1 & +y0 & -y1, lattice_mod)

# Set assemblies in their respective universes
u_ = mcdc.universe([assembly_uo2])
m_ = mcdc.universe([assembly_mox])
w_ = mcdc.universe([assembly_mod])

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
    +x0_ & -x1_ & +y0_ & -y1_,
    lattice_core,
    translation=[pitch * 17 * 3 / 2, -pitch * 17 * 3 / 2, 0.0],
)

# Root universe
mcdc.universe([core], root=True)

# =============================================================================
# iQMC Parameters
# =============================================================================
N = 1e4
Nx = 17 * 3 * 2
Ny = 17 * 3 * 2
G = 7
x_grid = np.linspace(0.0, pitch * 17 * 3, Nx + 1)
y_grid = np.linspace(-pitch * 17 * 3, 0.0, Ny + 1)

phi0 = np.ones((G, Nx, Ny))

mcdc.iQMC(
    x=x_grid,
    y=y_grid,
    g=np.ones(G),
    phi0=phi0,
    mode="batched",
    scores=["source-x", "source-y"],
)

# =============================================================================
# run mcdc
# =============================================================================

# Setting
mcdc.setting(N_particle=N)
mcdc.eigenmode(N_inactive=40, N_active=10)

# Run
mcdc.run()
