import numpy as np
import sys, os

# Get path to mcdc (not necessary if mcdc is installed)
sys.path.append('../../../')

import mcdc

# =============================================================================
# Set cells
# =============================================================================

# Set material XS
with np.load('SHEM-361.npz') as data:
    SigmaC = data['SigmaC']   # /cm
    SigmaS = data['SigmaS']
    SigmaF = data['SigmaF']
    nu_p   = data['nu_p']
    nu_d   = data['nu_d']
    chi_p  = data['chi_p']
    chi_d  = data['chi_d']
    G      = data['G']

SigmaC *= 2.0

# Set material
M = mcdc.Material(capture=SigmaC, scatter=SigmaS, fission=SigmaF, nu_p=nu_p,
                  chi_p=chi_p, nu_d=nu_d, chi_d=chi_d)

# Set surfaces
S0 = mcdc.SurfacePlaneX(-1E10,"reflective")
S1 = mcdc.SurfacePlaneX(1E10,"reflective")

# Set cells
C = mcdc.Cell([+S0, -S1], M)
cells = [C]

# =============================================================================
# Set source
# =============================================================================

Src = mcdc.SourceSimple(group=mcdc.DistUniformInt(0,G))
sources = [Src]

# =============================================================================
# Set problem and tally, and then run mcdc
# =============================================================================

mcdc.set_problem(cells, sources, N_hist=1E3)
mcdc.run()
