import numpy as np
import sys

import mcdc

N_particle = 1000 #int(sys.argv[2])
tag        = 'test' #sys.argv[3]

# =============================================================================
# Set model
# =============================================================================
# The infinite homogenous medium is modeled with reflecting slabs

# Load material data
with np.load('SHEM-361.npz') as data:
    SigmaT = data['SigmaT']   # /cm
    SigmaC = data['SigmaC']
    SigmaS = data['SigmaS']
    SigmaF = data['SigmaF']
    nu_p   = data['nu_p']
    nu_d   = data['nu_d']
    chi_p  = data['chi_p']
    chi_d  = data['chi_d']
    G      = data['G']
    v      = data['v']
    lamd   = data['lamd']
# Buckling and leakage XS to make the problem subcritical
R      = 10.0 # Sub
B_sq   = (np.pi/R)**2
D      = 1/(3*SigmaT)
SigmaL = D*B_sq
SigmaC += SigmaL

# Set material
m = mcdc.material(capture=SigmaC, scatter=SigmaS, fission=SigmaF, nu_p=nu_p,
                  chi_p=chi_p, nu_d=nu_d, chi_d=chi_d)

# Set surfaces
s1 = mcdc.surface('plane-x', x=-1E10, bc="reflective")
s2 = mcdc.surface('plane-x', x=1E10,  bc="reflective")

# Set cells
c = mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================
# Uniform in energy

source = mcdc.source(energy=np.ones(G))

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

# Tally
mcdc.tally(scores=['flux'])

# Setting
mcdc.setting(N_particle=N_particle, output='output_'+tag+'_'+str(N_particle), 
             progress_bar=False)

# Run
mcdc.run()
