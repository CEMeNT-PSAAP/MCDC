import h5py
import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# The infinite homogenous medium is modeled with reflecting slabs

# Load material data
lib = h5py.File('c5g7.h5', 'r')

# Materials
def set_mat(mat):
    return mcdc.material(capture = mat['capture'][:], 
                         scatter = mat['scatter'][:], 
                         fission = mat['fission'][:], 
                         nu_p    = mat['nu_p'][:], 
                         nu_d    = mat['nu_d'][:], 
                         chi_p   = mat['chi_p'][:], 
                         chi_d   = mat['chi_d'][:])
mat_uo2   = set_mat(lib['uo2'])
mat_mox43 = set_mat(lib['mox43'])
mat_mox7  = set_mat(lib['mox7'])
mat_mox87 = set_mat(lib['mox87'])
mat_gt    = set_mat(lib['gt'])
mat_fc    = set_mat(lib['fc'])
mat_mod   = set_mat(lib['mod'])

# Pin cells
pitch  = 1.26
radius = 0.54
# Surfaces
cy = mcdc.surface('cylinder-z', center=[0.0, 0.0], radius=radius)
# Cells
uo2  = mcdc.cell([-cy], mat_uo2)
mox4 = mcdc.cell([-cy], mat_mox43)
mox7 = mcdc.cell([-cy], mat_mox7)
mox8 = mcdc.cell([-cy], mat_mox87)
gt   = mcdc.cell([-cy], mat_gt)
fc   = mcdc.cell([-cy], mat_fc)
moo  = mcdc.cell([-cy], mat_mod)
mod  = mcdc.cell([+cy], mat_mod)
# Universes
u = mcdc.universe([uo2, mod])['ID']
l = mcdc.universe([mox4, mod])['ID']
m = mcdc.universe([mox7, mod])['ID']
n = mcdc.universe([mox8, mod])['ID']
g = mcdc.universe([gt, mod])['ID']
f = mcdc.universe([fc, mod])['ID']
w = mcdc.universe([moo, mod])['ID']
# Assembly lattice
grid = np.linspace(-pitch*17*3/2, pitch*17*3/2, 17*3+1)
mcdc.lattice(x=grid, y=grid, 
             universes=[[u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,g,u,u,g,u,u,g,u,u,u,u,u,l,m,m,m,m,g,m,m,g,m,m,g,m,m,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,g,u,u,u,u,u,u,u,u,u,g,u,u,u,l,m,m,g,m,n,n,n,n,n,n,n,m,g,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,m,m,m,n,n,n,n,n,n,n,n,n,m,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,g,u,u,g,u,u,g,u,u,g,u,u,g,u,u,l,m,g,n,n,g,n,n,g,n,n,g,n,n,g,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,m,m,n,n,n,n,n,n,n,n,n,n,n,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,m,m,n,n,n,n,n,n,n,n,n,n,n,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,g,u,u,g,u,u,f,u,u,g,u,u,g,u,u,l,m,g,n,n,g,n,n,f,n,n,g,n,n,g,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,m,m,n,n,n,n,n,n,n,n,n,n,n,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,m,m,n,n,n,n,n,n,n,n,n,n,n,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,g,u,u,g,u,u,g,u,u,g,u,u,g,u,u,l,m,g,n,n,g,n,n,g,n,n,g,n,n,g,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,m,m,m,n,n,n,n,n,n,n,n,n,m,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,g,u,u,u,u,u,u,u,u,u,g,u,u,u,l,m,m,g,m,n,n,n,n,n,n,n,m,g,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,g,u,u,g,u,u,g,u,u,u,u,u,l,m,m,m,m,g,m,m,g,m,m,g,m,m,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,m,m,g,m,m,g,m,m,g,m,m,m,m,l,u,u,u,u,u,g,u,u,g,u,u,g,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,g,m,n,n,n,n,n,n,n,m,g,m,m,l,u,u,u,g,u,u,u,u,u,u,u,u,u,g,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,m,n,n,n,n,n,n,n,n,n,m,m,m,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,g,n,n,g,n,n,g,n,n,g,n,n,g,m,l,u,u,g,u,u,g,u,u,g,u,u,g,u,u,g,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,n,n,n,n,n,n,n,n,n,n,n,m,m,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,n,n,n,n,n,n,n,n,n,n,n,m,m,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,g,n,n,g,n,n,f,n,n,g,n,n,g,m,l,u,u,g,u,u,g,u,u,f,u,u,g,u,u,g,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,n,n,n,n,n,n,n,n,n,n,n,m,m,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,n,n,n,n,n,n,n,n,n,n,n,m,m,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,g,n,n,g,n,n,g,n,n,g,n,n,g,m,l,u,u,g,u,u,g,u,u,g,u,u,g,u,u,g,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,m,n,n,n,n,n,n,n,n,n,m,m,m,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,g,m,n,n,n,n,n,n,n,m,g,m,m,l,u,u,u,g,u,u,u,u,u,u,u,u,u,g,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,m,m,g,m,m,g,m,m,g,m,m,m,m,l,u,u,u,u,u,g,u,u,g,u,u,g,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,m,m,m,m,m,m,m,m,m,m,m,m,m,m,m,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,u,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w],
                        [w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w,w]],
             bc_x_minus='reflective', bc_y_plus='reflective')

# =============================================================================
# Set source
# =============================================================================
# Uniform in energy

source = mcdc.source(x=[-pitch*17*2/2, pitch*17*2/2], y=[-pitch*17*2/2, pitch*17*2/2], 
                     energy=np.ones(7))

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

# Tally
mcdc.tally(scores=['fission'], x=grid, y=grid)

# Setting
mcdc.setting(N_hist=1E4)
mcdc.eigenmode(N_iter=20)
mcdc.population_control()

# Run
mcdc.run()
