import h5py
import numpy as np

import mcdc

# =============================================================================
# Materials
# =============================================================================

# Load material data
lib = h5py.File("c5g7_xs.h5", "r")


# Materials
def set_mat(mat):
    return mcdc.material(
        capture=mat["capture"][:],
        scatter=mat["scatter"][:],
        fission=mat["fission"][:],
        nu_p=mat["nu_p"][:],
        nu_d=mat["nu_d"][:],
        chi_p=mat["chi_p"][:],
        chi_d=mat["chi_d"][:],
        speed=mat["speed"],
        decay=mat["decay"],
    )


mat_uo2 = set_mat(lib["uo2"])
mat_mox43 = set_mat(lib["mox43"])
mat_mox7 = set_mat(lib["mox7"])
mat_mox87 = set_mat(lib["mox87"])
mat_gt = set_mat(lib["gt"])
mat_fc = set_mat(lib["fc"])
mat_cr = set_mat(lib["cr"])
mat_mod = set_mat(lib["mod"])

# =============================================================================
# Pin cells
# =============================================================================

pitch = 1.26
radius = 0.54

# Surfaces
cy = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=radius)
z0 = mcdc.surface("plane-z", z=-100.0)
z1 = mcdc.surface("plane-z", z=100.0)

# Cells
fc = mcdc.cell(-cy & +z0, mat_fc)  # Fission chamber (in and above core)
uo2 = mcdc.cell(-cy & +z0 & -z1, mat_uo2)  # Fuel rods (in core)
mox4 = mcdc.cell(-cy & +z0 & -z1, mat_mox43)
mox7 = mcdc.cell(-cy & +z0 & -z1, mat_mox7)
mox8 = mcdc.cell(-cy & +z0 & -z1, mat_mox87)
cr = mcdc.cell(-cy & +z1, mat_cr)  # Control rod (above core)
gti = mcdc.cell(-cy & +z0 & -z1, mat_gt)  # Guide tube (in core)
gta = mcdc.cell(-cy & +z1, mat_gt)  #            (above core)
mod = mcdc.cell(+cy, mat_mod)  # Moderator (outside rod)
modu = mcdc.cell(-cy & -z0, mat_mod)  #           (under rod)

# Universes (pin cells)
f = mcdc.universe([fc, mod, modu])
u = mcdc.universe([uo2, gta, mod, modu])
l = mcdc.universe([mox4, gta, mod, modu])
m = mcdc.universe([mox7, gta, mod, modu])
n = mcdc.universe([mox8, gta, mod, modu])
c = mcdc.universe([gti, cr, mod, modu])
g = mcdc.universe([gti, gta, mod, modu])

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

lattice_uo2_cr = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, c, u, u, c, u, u, c, u, u, u, u, u],
        [u, u, u, c, u, u, u, u, u, u, u, u, u, c, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, c, u, u, c, u, u, c, u, u, c, u, u, c, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, c, u, u, c, u, u, f, u, u, c, u, u, c, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, c, u, u, c, u, u, c, u, u, c, u, u, c, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, c, u, u, u, u, u, u, u, u, u, c, u, u, u],
        [u, u, u, u, u, c, u, u, c, u, u, c, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
        [u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u, u],
    ],
)

lattice_mox_cr = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, m, m, m, m, c, m, m, c, m, m, c, m, m, m, m, l],
        [l, m, m, c, m, n, n, n, n, n, n, n, m, c, m, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, c, n, n, c, n, n, c, n, n, c, n, n, c, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, c, n, n, c, n, n, f, n, n, c, n, n, c, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, m, n, n, n, n, n, n, n, n, n, n, n, m, m, l],
        [l, m, c, n, n, c, n, n, c, n, n, c, n, n, c, m, l],
        [l, m, m, m, n, n, n, n, n, n, n, n, n, m, m, m, l],
        [l, m, m, c, m, n, n, n, n, n, n, n, m, c, m, m, l],
        [l, m, m, m, m, c, m, m, c, m, m, c, m, m, m, m, l],
        [l, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, l],
        [l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l, l],
    ],
)

# Reflector
gt = mcdc.cell(-cy, mat_gt)  # Reflector
gto = mcdc.cell(+cy, mat_gt)  #
modi = mcdc.cell(-cy, mat_mod)  # Moderator (inside rod)
g = mcdc.universe([gt, gto])
w = mcdc.universe([modi, mod])
lattice_r = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [w, g, g, w, g, g, w, g, g, w, g, g, w, g, g, w, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, w, g, g, w, g, g, w, g, g, w, g, g, w, g, g, w],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, w, g, g, w, g, g, w, g, g, w, g, g, w, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [w, g, g, w, g, g, w, g, g, w, g, g, w, g, g, w, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, w, g, g, w, g, g, w, g, g, w, g, g, w, g, g, w],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
    ],
)

lattice_rll = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [w, g, g, g, w, g, g, g, w, g, g, w, w, w, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, w, w, w, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, w, w, w, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, w, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, w, g, g, g, g, g, g, w, g, g, g],
        [g, g, g, w, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, w, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, w, g, g, g, g, g, g, g, g, w, g, g, g],
        [g, g, g, g, g, g, g, w, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [w, g, g, g, g, g, g, g, g, g, g, g, g, w, g, g, g],
    ],
)

lattice_rul = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [w, g, g, g, g, g, g, g, g, g, g, g, g, w, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, w, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, w, g, g, g, g, g, g, g, g, w, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, w, g, g, g, g, g, g, g, g],
        [g, g, g, w, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, w, g, g, g, g, g, g, w, g, g, g],
        [g, g, g, g, g, g, g, g, g, w, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, w, w, w, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, w, w, w, g, g, g],
        [w, g, g, g, w, g, g, g, w, g, g, w, w, w, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
    ],
)

lattice_rur = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [g, g, g, w, g, g, g, g, g, g, g, g, g, g, g, g, w],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, w, g, g, g, g, g, g, g],
        [g, g, g, w, g, g, g, g, g, g, g, g, w, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, w, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, w, g, g, g],
        [g, g, g, w, g, g, g, g, g, g, w, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, w, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, w, w, w, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, w, w, w, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, w, w, w, g, g, w, g, g, g, w, g, g, g, w],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
    ],
)

lattice_rlr = mcdc.lattice(
    x=[-pitch * 17 / 2, pitch, 17],
    y=[-pitch * 17 / 2, pitch, 17],
    universes=[
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, w, w, w, g, g, w, g, g, g, w, g, g, g, w],
        [g, g, g, w, w, w, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, w, w, w, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, w, g, g, g, g, g, g, g, g, g],
        [g, g, g, w, g, g, g, g, g, g, w, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, w, g, g, g],
        [g, g, g, g, g, g, g, g, w, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, w, g, g, g, g, g, g, g, g, w, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, w, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g, g],
        [g, g, g, w, g, g, g, g, g, g, g, g, g, g, g, g, w],
    ],
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
assembly_rll = mcdc.cell(+x0 & -x1 & +y0 & -y1, lattice_rll)
assembly_rul = mcdc.cell(+x0 & -x1 & +y0 & -y1, lattice_rul)
assembly_rur = mcdc.cell(+x0 & -x1 & +y0 & -y1, lattice_rur)
assembly_rlr = mcdc.cell(+x0 & -x1 & +y0 & -y1, lattice_rlr)
assembly_r = mcdc.cell(+x0 & -x1 & +y0 & -y1, lattice_r)

# Set assemblies in their respective universes
u = mcdc.universe([assembly_uo2])
m = mcdc.universe([assembly_mox])
a = mcdc.universe([assembly_rll])
b = mcdc.universe([assembly_rul])
c = mcdc.universe([assembly_rur])
d = mcdc.universe([assembly_rlr])
e = mcdc.universe([assembly_r])

# =============================================================================
# Root universe: core
# =============================================================================

# Lattice
lattice_core = mcdc.lattice(
    x=[-pitch * 17 * 9 / 2, pitch * 17, 9],
    y=[-pitch * 17 * 9 / 2, pitch * 17, 9],
    universes=[
        [e, e, e, e, e, e, e, e, e],
        [e, b, b, u, m, u, c, c, e],
        [e, b, m, u, u, u, m, c, e],
        [e, u, u, m, u, m, u, u, e],
        [e, m, u, u, m, u, u, m, e],
        [e, u, u, m, u, m, u, u, e],
        [e, a, m, u, u, u, m, d, e],
        [e, e, a, u, m, u, d, d, e],
        [e, e, e, e, e, e, e, e, e],
    ],
)

# Core cell
# Surfaces
cy0 = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=pitch * 17 * 8.6 / 2)
cy1 = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=pitch * 17 * 9 / 2)
cy2 = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=pitch * 17 * 11 / 2)
cy3 = mcdc.surface(
    "cylinder-z", center=[0.0, 0.0], radius=pitch * 17 * 12 / 2, bc="vacuum"
)
zlo = mcdc.surface("plane-z", z=-130, bc="vacuum")
zup = mcdc.surface("plane-z", z=130, bc="vacuum")
# Cell
core = mcdc.cell(-cy0 & +zlo & -zup, lattice_core)
barrel = mcdc.cell(+cy0 & -cy1 & +zlo & -zup, mat_gt)
water = mcdc.cell(+cy1 & -cy2 & +zlo & -zup, mat_mod)
vessel = mcdc.cell(+cy2 & -cy3 & +zlo & -zup, mat_gt)

# Root universe
mcdc.universe([core, barrel, water, vessel], root=True)

# =============================================================================
# Set source
# =============================================================================
# Uniform in energy

source = mcdc.source(
    x=[-pitch * 17 * 7 / 2, pitch * 17 * 7 / 2],
    y=[-pitch * 17 * 7 / 2, pitch * 17 * 7 / 2],
    z=[-100.0, 100.0],
    energy=np.ones(7),
)

# =============================================================================
# Set tally and parameter, and then run mcdc
# =============================================================================

# Tally
x_grid = np.linspace(-pitch * 17 * 9 / 2, pitch * 17 * 9 / 2, 9 + 1)
y_grid = np.linspace(-pitch * 17 * 9 / 2, pitch * 17 * 9 / 2, 9 + 1)
z_grid = np.linspace(-130, 130, 14)
mcdc.tally.mesh_tally(scores=["flux"], x=x_grid, y=y_grid, z=z_grid, g="all")

# Setting
mcdc.setting(N_particle=30, census_bank_buff=2)
mcdc.eigenmode(N_inactive=1, N_active=2, gyration_radius="all")
mcdc.population_control()

# Run
mcdc.run()
