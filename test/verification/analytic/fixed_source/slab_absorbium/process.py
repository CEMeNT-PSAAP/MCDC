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
phi_ref, phi_x_ref, J_ref, J_x_ref = reference(x)

error    = []
error_x  = []
error_J  = []
error_Jx = []

for N_particle in N_particle_list:
    # Get results
    with h5py.File('output_%i.h5'%int(N_particle), 'r') as f:
        phi      = f['tally/flux/mean'][:]/dx
        phi_x    = f['tally/flux-x/mean'][:]
        J        = f['tally/current/mean'][:,0]/dx
        J_x      = f['tally/current-x/mean'][:,0]
    
    error.append(np.linalg.norm((phi - phi_ref)/phi_ref))
    error_x.append(np.linalg.norm((phi_x - phi_x_ref)/phi_x_ref))
    error_J.append(np.linalg.norm((J - J_ref)/J_ref))
    error_Jx.append(np.linalg.norm((J_x - J_x_ref)/J_x_ref))

plot_convergence('slab_absorbium_flux', N_particle_list, error)
plot_convergence('slab_absorbium_flux_x', N_particle_list, error_x)
plot_convergence('slab_absorbium_current', N_particle_list, error_J)
plot_convergence('slab_absorbium_current_x', N_particle_list, error_Jx)
