import numpy as np
import os
import mcdc
# Infinite medium of high-density polyethylene (HDPE)

# =============================================================================
# Import Cross Section Data
# =============================================================================
# G = 12 # G may equal 12, 70, or 618

# script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
# rel_path        = "./HDPE/"
# abs_file_path   = os.path.join(script_dir, rel_path)
# D = np.genfromtxt(abs_file_path+"D_{}G_HDPE.csv".format(G), delimiter=",")
# siga = np.genfromtxt(abs_file_path+"Siga_{}G_HDPE.csv".format(G), delimiter=",")
# sigs = np.genfromtxt(abs_file_path+"Scat_{}G_HDPE.csv".format(G), delimiter=",")

# =============================================================================
# Set Model
# =============================================================================

# x-bounds
LB = -5.0
RB = 5.0

# Set materials
# m1 = mcdc.material(capture=siga, scatter=sigs)

# Load material data
with np.load('../td_inf_shem361/SHEM-361.npz') as data:
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
R      = 45.0 # Sub
B_sq   = (np.pi/R)**2
D      = 1/(3*SigmaT)
SigmaL = D*B_sq
SigmaC += SigmaL

# Set material
m1 = mcdc.material(capture=SigmaC, scatter=SigmaS, fission=SigmaF, nu_p=nu_p,
                  chi_p=chi_p, nu_d=nu_d, chi_d=chi_d, speed=v, decay=lamd)



# Set surfaces
s1 = mcdc.surface('plane-x', x=LB, bc="reflective")
s2 = mcdc.surface('plane-x', x=RB, bc="reflective")

# Set cells
mcdc.cell([+s1, -s2], m1)



# =============================================================================
# iQMC Parameters
# =============================================================================

Nx             = 1
fixed_source   = np.zeros((G,Nx))
fixed_source[-1, :] = 1
fixed_source   /= fixed_source.sum()
material_idx   = np.zeros(Nx, dtype=int)
phi0           = np.random.uniform(size=(G,Nx))

mcdc.iQMC(g             = np.zeros((0,G)),
          x             = np.linspace(LB,RB,num=Nx+1), 
          fixed_source  = fixed_source, 
          phi0          = phi0, 
          material_idx  = material_idx, 
          maxitt        = 10,
          tol           = 1e-4,
          generator     = 'sobol')

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Setting
mcdc.setting(N_particle=2**14)

# Run
mcdc.run()
