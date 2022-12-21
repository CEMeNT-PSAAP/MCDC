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
    z   = f['tally/grid/z'][:]
    dz  = z[1:]-z[:-1]
    mu  = f['tally/grid/mu'][:]
    dmu = mu[1:]-mu[:-1]
phi_ref, phi_z_ref, J_ref, J_z_ref, psi_ref, psi_z_ref = reference(z,mu)

error       = []
error_z     = []
error_J     = []
error_Jz    = []
error_psi   = []
error_psi_z = []

for N_particle in N_particle_list:
    # Get results
    with h5py.File('output_%i.h5'%int(N_particle), 'r') as f:
        psi      = f['tally/flux/mean'][:]
        psi_z    = f['tally/flux-z/mean'][:]
        J        = f['tally/current/mean'][:,2]
        J_z      = f['tally/current-z/mean'][:,2]
    I = len(z)-1
    N = len(mu)-1

    # Scalar flux
    phi      = np.zeros(I)
    phi_z    = np.zeros(I+1)
    for i in range(I):
        phi[i]      += np.sum(psi[i,:])
        phi_z[i]    += np.sum(psi_z[i,:])
    phi_z[I]    += np.sum(psi_z[I,:])

    # Normalize
    phi    /= dz
    J      /= dz
    for n in range(N):
        psi[:,n]    = psi[:,n]/dz/dmu[n]
    
    error.append(np.linalg.norm((phi - phi_ref)/phi_ref))
    error_z.append(np.linalg.norm((phi_z - phi_z_ref)/phi_z_ref))
    error_J.append(np.linalg.norm((J - J_ref)/J_ref))
    error_Jz.append(np.linalg.norm((J_z - J_z_ref)/J_z_ref))
    error.append(np.linalg.norm((phi - phi_ref)/phi_ref))
    error_z.append(np.linalg.norm((phi_z - phi_z_ref)/phi_z_ref))

plot_convergence('slab_absorbium_flux', N_particle_list, error)
plot_convergence('slab_absorbium_flux_z', N_particle_list, error_z)
plot_convergence('slab_absorbium_current', N_particle_list, error_J)
plot_convergence('slab_absorbium_current_z', N_particle_list, error_Jz)
plot_convergence('slab_absorbium_angular_flux', N_particle_list, error_psi)
plot_convergence('slab_absorbium_angular_flux_z', N_particle_list, error_psi_z)
