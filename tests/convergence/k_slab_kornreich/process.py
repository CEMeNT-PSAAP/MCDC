import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Reference solution
# =============================================================================

x  = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.35, 1.5, 1.6, 
              1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5])
dx = x[1:] - x[:-1]
x_mid = 0.5*(x[1:] + x[:-1])
k_ref = 1.28657
phi_ref = np.array([1, 1.417721, 1.698988, 1.903163, 2.03435, 2.092069, 
                      2.075541, 1.984535, 1.818753, 1.574144, 1.199995, 
                      0.9532296, 0.7980474, 0.6788441, 0.5823852, 0.5020479, 
                      0.4337639, 0.3747058, 0.3226636, 0.2755115, 0.228371])
tmp = 0.5*(phi_ref[1:] + phi_ref[:-1])
phi_ref /= np.sum(tmp*dx)

# =============================================================================
# Plot results
# =============================================================================

error   = []
error_k = []
N_active_list = np.logspace(2, 3, 11)

# Results
with h5py.File('output.h5', 'r') as f:
    phi = f['tally/flux-x/mean'][:]
    k   = f['keff'][:]
    
# Get average and stdv over the active iterations
N_passive = 50

for N_active in N_active_list:
    N_active = int(N_active)
    print(N_active)
    phi_avg = np.zeros_like(phi[0])
    k_avg = 0.0
    for i in range(N_passive, N_active): 
        phi_avg += phi[i][:]
        k_avg += k[i]
    phi_avg /= N_active    
    tmp = 0.5*(phi_avg[1:] + phi_avg[:-1])
    norm = np.sum(tmp*dx)
    phi_avg /= norm
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
