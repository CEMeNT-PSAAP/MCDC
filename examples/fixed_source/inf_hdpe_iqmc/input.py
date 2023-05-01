import numpy as np
import os
import mcdc

# Infinite medium of high-density polyethylene (HDPE)

# =============================================================================
# Import Cross Section Data
# =============================================================================
G = 12  # G must be 12, 70, or 618

script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
rel_path = "./HDPE/"
abs_file_path = os.path.join(script_dir, rel_path)
D = np.genfromtxt(abs_file_path + "D_{}G_HDPE.csv".format(G), delimiter=",")
SigmaA = np.genfromtxt(abs_file_path + "Siga_{}G_HDPE.csv".format(G), delimiter=",")
SigmaS = np.genfromtxt(abs_file_path + "Scat_{}G_HDPE.csv".format(G), delimiter=",")
SigmaS = np.flip(SigmaS, 1)

# =============================================================================
# Set Model
# =============================================================================

# x-bounds
LB = 0.0
RB = 5.0
# Set material
m1 = mcdc.material(capture=SigmaA, scatter=SigmaS)

# Set surfaces
s1 = mcdc.surface("plane-x", x=LB, bc="reflective")
s2 = mcdc.surface("plane-x", x=RB, bc="reflective")

# Set cells
mcdc.cell([+s1, -s2], m1)


# =============================================================================
# iQMC Parameters
# =============================================================================

Nx = 5
fixed_source = np.ones((G, Nx))
# material_idx = np.zeros(Nx, dtype=int)
phi0 = np.ones((G, Nx))

mcdc.iQMC(
    g=np.ones((0, G)),
    x=np.linspace(LB, RB, num=Nx + 1),
    fixed_source=fixed_source,
    phi0=phi0,
    maxitt=25,
    tol=1e-3,
    generator="halton",
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# weight roulette
chance = 0.9
threshold = 1e-3
mcdc.weight_roulette(chance, threshold)
# Setting
mcdc.setting(N_particle=1e3)

# Run
mcdc.run()
