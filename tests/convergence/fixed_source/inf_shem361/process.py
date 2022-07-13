import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys


# =============================================================================
# Reference solution
# =============================================================================

# Load XS
with np.load('SHEM-361.npz') as data:
    data     = np.load('SHEM-361.npz')
    SigmaT   = data['SigmaT'] + data['SigmaC']
    nuSigmaF = data['nuSigmaF']
    SigmaS   = data['SigmaS']
    E        = data['E']

G = len(SigmaT)
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

A = np.diag(SigmaT) - SigmaS - nuSigmaF
Q = np.ones(G)/G

phi_ref = np.linalg.solve(A,Q)*E_mid/dE

# =============================================================================
# Plot results
# =============================================================================

error   = []
N_max = int(sys.argv[1])
N_particle_list = np.logspace(3, N_max, (N_max-3)*2+1)

for N_particle in N_particle_list:
    # Results
    with h5py.File('output_%i.h5'%int(N_particle), 'r') as f:
        phi = f['tally/flux/mean'][:]*E_mid/dE
    
    error.append(np.linalg.norm((phi - phi_ref)/phi_ref))

line = 1.0/np.sqrt(N_particle_list)
line *= error[N_max-3]/line[N_max-3]
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

