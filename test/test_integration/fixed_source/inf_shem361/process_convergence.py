import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

N_min = int(sys.argv[1])
N_max = int(sys.argv[2])

# =============================================================================
# Reference solution
# =============================================================================

# Load material data
with np.load('SHEM-361.npz') as data:
    SigmaT = data['SigmaT']   # /cm
    SigmaS = data['SigmaS']
    nuSigmaF = data['nuSigmaF']
    G      = data['G']
    E      = data['E']
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

# Buckling and leakage XS to make the problem subcritical
R      = 10.0 # Sub
B_sq   = (np.pi/R)**2
D      = 1/(3*SigmaT)
SigmaL = D*B_sq
SigmaT += SigmaL

A = np.diag(SigmaT) - SigmaS - nuSigmaF
Q = np.ones(G)/G

phi_ref = np.linalg.solve(A,Q)*E_mid/dE

# =============================================================================
# Plot results
# =============================================================================

error   = []
N_particle_list = np.logspace(N_min, N_max, (N_max-N_min)*2+1)

for N_particle in N_particle_list:
    # Results
    with h5py.File('output_convergence_%i.h5'%int(N_particle), 'r') as f:
        phi = f['tally/flux/mean'][:]*E_mid/dE
    
    error.append(np.linalg.norm((phi - phi_ref)/phi_ref))

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
plt.title('flux')
plt.savefig('flux.png')
plt.clf()

