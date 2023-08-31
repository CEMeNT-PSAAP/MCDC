from plotter import plot_convergence
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
phi_ref = data["phi"].T
n_ref = data["n"].T

# Error container
error = np.zeros(len(N_particle_list))
error_n = np.zeros(len(N_particle_list))

error_max = np.zeros(len(N_particle_list))
error_max_n = np.zeros(len(N_particle_list))

# Calculate error
for k, N_particle in enumerate(N_particle_list):
    # Get results
    with h5py.File("output_%i.h5" % int(N_particle), "r") as f:
        phi = f["tally/flux-t/mean"][:]
    phi = phi[:, 1:]
    with np.load("SHEM-361.npz") as data:
        speed = data["v"]

    # Neutron density
    n = np.zeros(K)
    for k in range(K):
        n[k] = np.sum(phi[:, k] / speed)

    error[k] = tool.error(phi, phi_ref)
    error_n[k] = tool.error(n, n_ref)
    error_max[k] = tool.error_max(phi, phi_ref)
    error_max_n[k] = tool.error_max(n, n_ref)

tool.plot_convergence("inf_shem361_td_flux", N_particle_list, error, error_max)
tool.plot_convergence("inf_shem361_td_n", N_particle_list, error_n, error_max_n)
