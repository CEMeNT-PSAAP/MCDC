import h5py, sys
import numpy as np

import mcdc


# Get IC generator parameters
with h5py.File("output_k.h5", "r") as f:
    neutron_density = f["global_tally/neutron/mean"][()]
    max_neutron_density = f["global_tally/neutron/max"][()]
    precursor_density = f["global_tally/precursor/mean"][()]
    max_precursor_density = f["global_tally/precursor/max"][()]
    keff = f["k_mean"][()]

# =============================================================================
# Materials
# =============================================================================

# Load material data
lib = h5py.File("c5g7_xs.h5", "r")


# Setter
def set_mat(mat):
    return mcdc.material(
        capture=mat["capture"][:],
        scatter=mat["scatter"][:],
        fission=mat["fission"][:],
        nu_p=mat["nu_p"][:] / keff,
        nu_d=mat["nu_d"][:] / keff,
        chi_p=mat["chi_p"][:],
        chi_d=mat["chi_d"][:],
        speed=mat["speed"],
        decay=mat["decay"],
    )


mat_uo2 = set_mat(lib["uo2"])  # Fuel: UO2
mat_mox43 = set_mat(lib["mox43"])  # Fuel: MOX 4.3%
mat_mox7 = set_mat(lib["mox7"])  # Fuel: MOX 7.0%
mat_mox87 = set_mat(lib["mox87"])  # Fuel: MOX 8.7%
mat_gt = set_mat(lib["gt"])  # Guide tube
mat_fc = set_mat(lib["fc"])  # Fission chamber
mat_cr = set_mat(lib["cr"])  # Control rod
mat_mod = set_mat(lib["mod"])  # Moderator

# =============================================================================
# Pin cells
# =============================================================================

pitch = 1.26
radius = 0.54
core_height = 128.52
refl_thick = 21.42

# Control rod banks fractions
#   All out: 0.0
#   All in : 1.0
cr1 = 0.0
cr2 = 0.0
cr3 = 0.0
cr4 = 0.0
# Control rod banks interfaces
cr1 = core_height * (0.5 - cr1)
cr2 = core_height * (0.5 - cr2)
cr3 = core_height * (0.5 - cr3)
cr4 = core_height * (0.5 - cr4)

# Surfaces
cy = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=radius)
z1 = mcdc.surface("plane-z", z=cr1)  # Control rod banks interfaces
z2 = mcdc.surface("plane-z", z=cr2)
z3 = mcdc.surface("plane-z", z=cr3)
z4 = mcdc.surface("plane-z", z=cr4)
zf = mcdc.surface("plane-z", z=core_height / 2)

# Fission chamber
fc = mcdc.cell([-cy], mat_fc)
mod = mcdc.cell([+cy], mat_mod)
fission_chamber = mcdc.universe([fc, mod])["ID"]

# Fuel rods
uo2 = mcdc.cell([-cy, -zf], mat_uo2)
mox4 = mcdc.cell([-cy, -zf], mat_mox43)
mox7 = mcdc.cell([-cy, -zf], mat_mox7)
mox8 = mcdc.cell([-cy, -zf], mat_mox87)
moda = mcdc.cell([-cy, +zf], mat_mod)  # Water above pin
fuel_uo2 = mcdc.universe([uo2, mod, moda])["ID"]
fuel_mox43 = mcdc.universe([mox4, mod, moda])["ID"]
fuel_mox7 = mcdc.universe([mox7, mod, moda])["ID"]
fuel_mox87 = mcdc.universe([mox8, mod, moda])["ID"]

# Control rods and guide tubes
cr1 = mcdc.cell([-cy, +z1], mat_cr)
cr2 = mcdc.cell([-cy, +z2], mat_cr)
cr3 = mcdc.cell([-cy, +z3], mat_cr)
cr4 = mcdc.cell([-cy, +z4], mat_cr)
gt1 = mcdc.cell([-cy, -z1], mat_gt)
gt2 = mcdc.cell([-cy, -z2], mat_gt)
gt3 = mcdc.cell([-cy, -z3], mat_gt)
gt4 = mcdc.cell([-cy, -z4], mat_gt)
control_rod1 = mcdc.universe([cr1, gt1, mod])["ID"]
control_rod2 = mcdc.universe([cr2, gt2, mod])["ID"]
control_rod3 = mcdc.universe([cr3, gt3, mod])["ID"]
control_rod4 = mcdc.universe([cr4, gt4, mod])["ID"]

# =============================================================================
# Fuel lattices
# =============================================================================

# UO2 lattice 1
u = fuel_uo2
c = control_rod1
f = fission_chamber
lattice_1 = mcdc.lattice(
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

# MOX lattice 2
l = fuel_mox43
m = fuel_mox7
n = fuel_mox87
c = control_rod2
f = fission_chamber
lattice_2 = mcdc.lattice(
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

# MOX lattice 3
l = fuel_mox43
m = fuel_mox7
n = fuel_mox87
c = control_rod3
f = fission_chamber
lattice_3 = mcdc.lattice(
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

# UO2 lattice 4
u = fuel_uo2
c = control_rod4
f = fission_chamber
lattice_4 = mcdc.lattice(
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

# =============================================================================
# Assemblies and core
# =============================================================================

# Surfaces
x0 = mcdc.surface("plane-x", x=0.0, bc="reflective")
x1 = mcdc.surface("plane-x", x=pitch * 17)
x2 = mcdc.surface("plane-x", x=pitch * 17 * 2)
x3 = mcdc.surface("plane-x", x=pitch * 17 * 3, bc="vacuum")

y0 = mcdc.surface("plane-y", y=-pitch * 17 * 3, bc="vacuum")
y1 = mcdc.surface("plane-y", y=-pitch * 17 * 2)
y2 = mcdc.surface("plane-y", y=-pitch * 17)
y3 = mcdc.surface("plane-y", y=0.0, bc="reflective")

z0 = mcdc.surface("plane-z", z=-(core_height / 2 + refl_thick), bc="vacuum")
z1 = mcdc.surface("plane-z", z=-(core_height / 2))
z2 = mcdc.surface("plane-z", z=(core_height / 2 + refl_thick), bc="vacuum")

# Assembly cells
center = np.array([pitch * 17 / 2, -pitch * 17 / 2, 0.0])
assembly_1 = mcdc.cell([+x0, -x1, +y2, -y3, +z1, -z2], lattice_1, lattice_center=center)

center += np.array([pitch * 17, 0.0, 0.0])
assembly_2 = mcdc.cell([+x1, -x2, +y2, -y3, +z1, -z2], lattice_2, lattice_center=center)

center += np.array([-pitch * 17, -pitch * 17, 0.0])
assembly_3 = mcdc.cell([+x0, -x1, +y1, -y2, +z1, -z2], lattice_3, lattice_center=center)

center += np.array([pitch * 17, 0.0, 0.0])
assembly_4 = mcdc.cell([+x1, -x2, +y1, -y2, +z1, -z2], lattice_4, lattice_center=center)

# Bottom reflector cell
reflector_bottom = mcdc.cell([+x0, -x3, +y0, -y3, +z0, -z1], mat_mod)

# Side reflectors
reflector_south = mcdc.cell([+x0, -x3, +y0, -y1, +z1, -z2], mat_mod)
reflector_east = mcdc.cell([+x2, -x3, +y1, -y3, +z1, -z2], mat_mod)

# Root universe
mcdc.universe(
    [
        assembly_1,
        assembly_2,
        assembly_3,
        assembly_4,
        reflector_bottom,
        reflector_south,
        reflector_east,
    ],
    root=True,
)

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

# Setting
mcdc.setting(source_file="output_k.h5", output_name="output_ic")
mcdc.IC_generator(
    N_neutron=1e6,
    N_precursor=1e6,
    neutron_density=neutron_density,
    max_neutron_density=max_neutron_density,
    precursor_density=precursor_density,
    max_precursor_density=max_precursor_density,
)

# Run
mcdc.run()
