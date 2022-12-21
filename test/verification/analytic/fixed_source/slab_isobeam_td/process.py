import numpy as np
import h5py
import sys
sys.path.append('../../util')

from reference import reference
from plotter   import plot_convergence


N_min = int(sys.argv[1])
N_max = int(sys.argv[2])
N_particle_list = np.logspace(N_min, N_max, (N_max-N_min)*2+1)

# Reference solution 
with h5py.File('output_%i.h5'%int(N_particle_list[0]), 'r') as f:
    x  = f['tally/grid/x'][:]
    t  = f['tally/grid/t'][:]
    K  = len(t)-1
    dx = x[1:]-x[:-1]
phi_t_ref = reference(x,t)

error_t  = []

for N_particle in N_particle_list:
    # Get results
    with h5py.File('output_%i.h5'%int(N_particle), 'r') as f:
        phi_t = f['tally/flux-t/mean'][1:]
    for k in range(K):
        phi_t[k] *= (0.5/dx)
    error_t.append(np.linalg.norm(phi_t - phi_t_ref))

plot_convergence('slab_isoBeam_td_flux_t', N_particle_list, error_t)
