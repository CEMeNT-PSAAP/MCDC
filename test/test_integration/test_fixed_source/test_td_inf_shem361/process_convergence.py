import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.linalg import expm
import matplotlib.animation as animation
import sys

N_min = int(sys.argv[1])
N_max = int(sys.argv[2])

# =============================================================================
# Reference solution
# =============================================================================

# Load XS
with np.load('SHEM-361.npz') as data:
    SigmaT     = data['SigmaT']
    SigmaC     = data['SigmaC']
    SigmaS     = data['SigmaS']
    nuSigmaF_p = data['nuSigmaF_p']
    SigmaF     = data['SigmaF']
    nu_p       = data['nu_p']
    nu_d       = data['nu_d']
    chi_p      = data['chi_p']
    chi_d      = data['chi_d']
    G          = data['G']
    J          = data['J']
    E          = data['E']
    v          = data['v']
    lamd       = data['lamd']
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]
# Buckling and leakage XS
R      = 45.0 # Sub
B_sq   = (np.pi/R)**2
D      = 1/(3*SigmaT)
SigmaL = D*B_sq
SigmaT += SigmaL

# Time grid
N = 100
t = np.logspace(-10,2,N)

# Matrix and RHS source
A = np.zeros([G+J, G+J])
Q = np.zeros(G+J)

# Top-left [GxG]: phi --> phi
A[:G,:G] = SigmaS + nuSigmaF_p - np.diag(SigmaT)

# Top-right [GxJ]: C --> phi
A[:G,G:] = np.multiply(chi_d,lamd)

# Bottom-left [JxG]: phi --> C
A[G:,:G] = np.multiply(nu_d,SigmaF)

# bottom-right [JxJ]: C --> C
A[G:,G:] = -np.diag(lamd)

# Multiply with neutron speed
AV       = np.copy(A)
AV[:G,:] = np.dot(np.diag(v), A[:G,:])

# Initial condition
PHI_init = np.zeros(G+J)
#PHI_init[:G] = v/G
PHI_init[G-2] = v[-2]

# Analytical particular solution
PHI_p = np.dot(np.linalg.inv(-A),Q)

# Allocate solution
PHI   = np.zeros([N,G+J])
C     = np.zeros((N,J))
n_ref = np.zeros(N)

# Analytical solution
for n in range(N):
    # Flux
    PHI_h    = np.dot(expm(AV*t[n]),(PHI_init - PHI_p))
    PHI[n,:] = PHI_h + PHI_p
    C[n,:]   = PHI[n,G:]
    n_ref[n] = sum(np.divide(PHI[n,:G],v))
phi_ref = PHI[:,:G]

# =============================================================================
# Plot results
# =============================================================================

error   = []
error_n = []
N_particle_list = np.logspace(N_min, N_max, (N_max-N_min)*2+1)

for N_particle in N_particle_list:
    with h5py.File('output_convergence_%i.h5'%int(N_particle), 'r') as f:
        phi = f['tally/flux-t/mean'][:]
    n_tot = np.zeros_like(n_ref)
    for n in range(N):
        n_tot[n] = sum(np.divide(phi[:,n+1],v))
    phi = phi[:,1:]

    phi = np.transpose(phi)
    n_tot = np.transpose(n_tot)
   
    error.append(np.linalg.norm((phi - phi_ref)/phi_ref))
    error_n.append(np.linalg.norm((n_tot - n_ref)/n_ref))

line = 1.0/np.sqrt(N_particle_list)
line *= error[N_max-N_min]/line[N_max-N_min]
plt.plot(N_particle_list, error, 'bo', fillstyle='none')
plt.plot(N_particle_list, line, 'r--', label=r'$N^{-0.5}$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('2-norm of relative error')
plt.xlabel(r'# of histories, $N$')
plt.legend()
plt.grid()
plt.title('flux_t')
plt.savefig('flux_t.png')
plt.clf()

line = 1.0/np.sqrt(N_particle_list)
line *= error_n[N_max-N_min]/line[N_max-N_min]
plt.plot(N_particle_list, error_n, 'bo', fillstyle='none')
plt.plot(N_particle_list, line, 'r--', label=r'$N^{-0.5}$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('2-norm of relative error')
plt.xlabel(r'# of histories, $N$')
plt.legend()
plt.grid()
plt.title('n_t')
plt.savefig('n_t.png')
plt.clf()
