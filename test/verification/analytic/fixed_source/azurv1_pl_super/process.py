import numpy as np
import h5py
import sys

sys.path.append("../../")
import tool


# Cases run
N_min = int(sys.argv[1])
N_max = int(sys.argv[2])
N_particle_list = np.logspace(N_min, N_max, (N_max - N_min) * 2 + 1)

# Reference solution
data = np.load("reference.npz")
phi_x_ref = data["phi_x"]
phi_t_ref = data["phi_t"]
phi_ref = data["phi"]

# Error containers
error = np.zeros(len(N_particle_list))
error_x = np.zeros(len(N_particle_list))
error_t = np.zeros(len(N_particle_list))

error_max = np.zeros(len(N_particle_list))
error_max_x = np.zeros(len(N_particle_list))
error_max_t = np.zeros(len(N_particle_list))

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
        phi_x = f["tally/flux-x/mean"][:]
        phi_t = f["tally/flux-t/mean"][:]

    # Normalize
    for k in range(K):
        phi[k] /= dx * dt[k]
        phi_x[k] /= dt[k]
        phi_t[k] /= dx
    phi_t[K] /= dx
    phi_t = phi_t[1:]

    # Get error
    error[i] = tool.error(phi, phi_ref)
    error_x[i] = tool.error(phi_x, phi_x_ref)
    error_t[i] = tool.error(phi_t, phi_t_ref)

    error_max[i] = tool.error_max(phi, phi_ref)
    error_max_x[i] = tool.error_max(phi_x, phi_x_ref)
    error_max_t[i] = tool.error_max(phi_t, phi_t_ref)

# Plot
tool.plot_convergence("azurv1_pl_flux", N_particle_list, error, error_max)
tool.plot_convergence("azurv1_pl_flux_x", N_particle_list, error_x, error_max_x)
tool.plot_convergence("azurv1_pl_flux_t", N_particle_list, error_t, error_max_t)
