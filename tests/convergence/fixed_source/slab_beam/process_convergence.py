import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

N_min = int(sys.argv[1])
N_max = int(sys.argv[2])

# =============================================================================
# Reference solution
# =============================================================================

# Load grids
with h5py.File('output_convergence_1000.h5', 'r') as f:
    x = f['tally/grid/x'][:]
dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])

# XS
SigmaT1 = 1.0
SigmaT2 = 1.5
SigmaT3 = 2.0

# Spatial average flux
phi_ref = -(1-np.exp(SigmaT2*dx))*np.exp(-SigmaT2*x[1:21])/SigmaT2/dx
phi_ref = np.append(phi_ref, np.exp(-SigmaT2*2.0)*-(1-np.exp(SigmaT3*dx))*np.exp(-SigmaT3*x[1:21])/SigmaT3/dx)
phi_ref = np.append(phi_ref, np.exp(-(SigmaT2+SigmaT3)*2.0)*-(1-np.exp(SigmaT1*dx))*np.exp(-SigmaT1*x[1:21])/SigmaT1/dx)

# Spatial grid flux
phi_x_ref = np.exp(-SigmaT2*x[1:21])
phi_x_ref = np.append(phi_x_ref, np.exp(-SigmaT2*2.0)*np.exp(-SigmaT3*x[1:21]))
phi_x_ref = np.append(phi_x_ref, np.exp(-SigmaT2*2.0)*np.exp(-SigmaT3*2.0)*np.exp(-SigmaT1*x[1:21]))

# =============================================================================
# Plot results
# =============================================================================

error   = []
error_x = []
N_particle_list = np.logspace(N_min, N_max, (N_max-N_min)*2+1)

for N_particle in N_particle_list:
    # Get results
    with h5py.File('output_convergence_%i.h5'%int(N_particle), 'r') as f:
        phi      = f['tally/flux/mean'][:]/dx
        phi_x    = f['tally/flux-x/mean'][1:]
    
    error.append(np.linalg.norm((phi - phi_ref)/phi_ref))
    error_x.append(np.linalg.norm((phi_x - phi_x_ref)/phi_x_ref))

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
