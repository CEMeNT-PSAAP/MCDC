import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

# Load XS
with np.load('SHEM-361.npz') as data:
    data     = np.load('SHEM-361.npz')
    SigmaT   = data['SigmaT']
    nuSigmaF = data['nuSigmaF']
    SigmaS   = data['SigmaS']
    E        = data['E']

G = len(SigmaT)
E_mid = 0.5*(E[1:]+E[:-1])
dE    = E[1:]-E[:-1]

A = np.dot(np.linalg.inv(np.diag(SigmaT) - SigmaS),nuSigmaF)
w,v = np.linalg.eig(A)
idx = w.argsort()[::-1]   
w = w[idx]
v = v[:,idx]
k_ref = w[0]
phi_ref = v[:,0]
if phi_ref[0] < 0.0:
    phi_ref *= -1
phi_ref[:] /= np.sum(phi_ref[:])
phi_ref = phi_ref/dE*E_mid

# =============================================================================
# Plot results
# =============================================================================

error   = []
error_k = []
N_active_list = np.logspace(2, 3, 11)

# Results
with h5py.File('output.h5', 'r') as f:
    phi    = f['tally/flux/mean'][:]
    k      = f['keff'][:]

# Get average and stdv over the active iterations
N_passive = 50

for N_active in N_active_list:
    N_active = int(N_active)
    phi_avg = np.zeros_like(phi[0])
    k_avg = 0.0
    for i in range(N_passive, N_active+N_passive): 
        phi_avg += phi[i][:]
        k_avg += k[i]
    phi_avg /= N_active    
    norm = np.sum(phi_avg)
    phi_avg = phi_avg/norm/dE*E_mid
    k_avg /= N_active    
    
    error.append(np.linalg.norm((phi_avg - phi_ref)/phi_ref))
    error_k.append(np.linalg.norm((k_avg - k_ref)/k_ref))

line = 1.0/np.sqrt(N_active_list)
line *= error[5]/line[5]
plt.plot(N_active_list, error, 'bo', fillstyle='none')
plt.plot(N_active_list, line, 'r--', label=r'$N^{-0.5}$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('2-norm of relative error')
plt.xlabel(r'# of histories, $N$')
plt.legend()
plt.grid()
plt.title('flux')
plt.savefig('flux.png')
plt.clf()

line = 1.0/np.sqrt(N_active_list)
line *= error_k[5]/line[5]
plt.plot(N_active_list, error_k, 'bo', fillstyle='none')
plt.plot(N_active_list, line, 'r--', label=r'$N^{-0.5}$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('2-norm of relative error')
plt.xlabel(r'# of histories, $N$')
plt.legend()
plt.grid()
plt.title('keff')
plt.savefig('keff.png')
plt.clf()
