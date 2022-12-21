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
    dx = x[1:]-x[:-1]
phi_ref, phi_x_ref = reference(x)

error    = []
error_x  = []

for N_particle in N_particle_list:
    # Get results
    with h5py.File('output_%i.h5'%int(N_particle), 'r') as f:
        phi      = f['tally/flux/mean'][:]/dx*101.
        phi_x    = f['tally/flux-x/mean'][:]*101.
    
    error.append(np.linalg.norm((phi - phi_ref)/phi_ref))
    error_x.append(np.linalg.norm((phi_x - phi_x_ref)/phi_x_ref))

plot_convergence('reed_flux', N_particle_list, error)
plot_convergence('reed_flux_x', N_particle_list, error_x)
