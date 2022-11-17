import numpy as np
import h5py
import sys
sys.path.append('../../util')

from plotter   import plot_convergence


N_min = int(sys.argv[1])
N_max = int(sys.argv[2])
N_particle_list = np.logspace(N_min, N_max, (N_max-N_min)*2+1)

# Reference solution 
data = np.load('azurv1_pl.npz')
phi_x_ref = data['phi_x']
phi_t_ref = data['phi_t']
phi_ref = data['phi']

error    = []
error_x  = []
error_t  = []

for N_particle in N_particle_list:
    # Get results
    with h5py.File('output_%i.h5'%int(N_particle), 'r') as f:
        x     = f['tally/grid/x'][:]
        dx    = x[1:]-x[:-1]
        t     = f['tally/grid/t'][:]
        dt    = t[1:]-t[:-1]
        phi   = f['tally/flux/mean'][:]
        phi_x = f['tally/flux-x/mean'][:]
        phi_t = f['tally/flux-t/mean'][:]
        K = len(t) - 1
    for k in range(K):
        phi[k]      /= (dx*dt[k])
        phi_x[k]    /= (dt[k])
        phi_t[k]    /= (dx)
    phi_t[K] /= (dx)
    phi_t = phi_t[1:]
    
    error.append(np.linalg.norm(phi - phi_ref))
    error_x.append(np.linalg.norm(phi_x - phi_x_ref))
    error_t.append(np.linalg.norm(phi_t - phi_t_ref))

plot_convergence('azurv1_pl_flux', N_particle_list, error)
plot_convergence('azurv1_pl_flux_x', N_particle_list, error_x)
plot_convergence('azurv1_pl_flux_t', N_particle_list, error_t)
