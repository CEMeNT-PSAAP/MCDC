import numpy as np
import h5py
import sys

sys.path.append("../")
import tool


# Cases run
N_min = int(sys.argv[1])
N_max = int(sys.argv[2])
N = int(sys.argv[3])
N_particle_list = np.logspace(N_min, N_max, N)

# Reference solution
data = np.load("reference.npz")
phi_ref = data["phi"]

# Error containers
error = np.zeros(len(N_particle_list))

error_max = np.zeros(len(N_particle_list))

# Calculate error
for i, N_particle in enumerate(N_particle_list):
    # Get results
    with h5py.File("output_%i.h5" % int(N_particle), "r") as f:
        x = f["tally/grid/x"][:]
        dx = x[1:] - x[:-1]
        t = f["tally/grid/t"][:]
        dt = t[1:] - t[:-1]
        K = len(t) - 1

        phi = f["tally/flux/mean"][:]

    # Normalize
    for k in range(K):
        phi[k] /= dx * dt[k]

    # Get error
    error[i] = tool.error(phi, phi_ref)

    error_max[i] = tool.error_max(phi, phi_ref)

# Plot
tool.plot_convergence("azurv1_pl_flux", N_particle_list, error, error_max)
