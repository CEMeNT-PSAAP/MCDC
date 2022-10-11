import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

N_min = int(sys.argv[1])
N_max = int(sys.argv[2])

# =============================================================================
# Reference solution (SS)
# =============================================================================

# Load grids
with h5py.File('output_convergence_1000.h5', 'r') as f:
    x = f['tally/grid/x'][:]
    t = f['tally/grid/t'][:]

dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])
dt    = t[1:] - t[:-1]
K     = len(dt)
J     = len(x_mid)

data = np.load('azurv1_pl.npz')
phi_x_ref = data['phi_x']
phi_t_ref = data['phi_t']
phi_ref = data['phi']

# =============================================================================
# Plot results
# =============================================================================

error   = []
error_x = []
error_t = []
N_particle_list = np.logspace(N_min, N_max, (N_max-N_min)*2+1)

for N_particle in N_particle_list:
    with h5py.File('output_convergence_%i.h5'%int(N_particle), 'r') as f:
        phi      = f['tally/flux/mean'][:]
        phi_x    = f['tally/flux-x/mean'][:]
        phi_t    = f['tally/flux-t/mean'][:]
    for k in range(K):
        phi[k]      /= (dx*dt[k])
        phi_x[k]    /= (dt[k])
        phi_t[k]    /= (dx)
    phi_t[K] /= (dx)
    phi_t = phi_t[1:]
   
    error.append(np.linalg.norm(phi - phi_ref))
    error_x.append(np.linalg.norm(phi_x - phi_x_ref))
    error_t.append(np.linalg.norm(phi_t - phi_t_ref))

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

line = 1.0/np.sqrt(N_particle_list)
line *= error_x[N_max-N_min]/line[N_max-N_min]
plt.plot(N_particle_list, error_x, 'bo', fillstyle='none')
plt.plot(N_particle_list, line, 'r--', label=r'$N^{-0.5}$')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('2-norm of relative error')
plt.xlabel(r'# of histories, $N$')
plt.legend()
plt.grid()
plt.title('flux_x')
plt.savefig('flux_x.png')
plt.clf()

line = 1.0/np.sqrt(N_particle_list)
line *= error_t[N_max-N_min]/line[N_max-N_min]
plt.plot(N_particle_list, error_t, 'bo', fillstyle='none')
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
