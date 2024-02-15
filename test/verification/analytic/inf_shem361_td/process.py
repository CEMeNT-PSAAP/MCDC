import numpy as np
import h5py
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append("../")
import tool


# Cases run
N_min = int(sys.argv[1])
N_max = int(sys.argv[2])
N_particle_list = np.logspace(N_min, N_max, (N_max - N_min) * 2 + 1)

# Reference solution
data = np.load("reference.npz")
phi_ref_ = data["phi"].T
n_ref = data["n"].T

# Error container
error = np.zeros(len(N_particle_list))
error_n = np.zeros(len(N_particle_list))

error_max = np.zeros(len(N_particle_list))
error_max_n = np.zeros(len(N_particle_list))

# Calculate error
for i, N_particle in enumerate(N_particle_list):
    # Get results
    with np.load("SHEM-361.npz") as data:
        E = data["E"]
        G = data["G"]
        speed = data["v"]
        E_mid = 0.5 * (E[1:] + E[:-1])
        dE = E[1:] - E[:-1]
    with h5py.File("output_%i.h5" % int(N_particle), "r") as f:
        t = f["tally/grid/t"][:]
        dt = t[1:] - t[:-1]
        K = len(t) - 1
        phi = f["tally/flux/mean"][:]

    # Neutron density
    n = np.zeros(K)
    for k in range(K):
        n[k] = np.sum(phi[:, k] / speed) / dt[k]

    # Normalize
    phi_ref = np.zeros_like(phi_ref_)
    for k in range(K):
        phi_ref[:, k] = phi_ref_[:, k] * E_mid / dE
        phi[:, k] *= E_mid / dE / dt[k]

    error[i] = tool.rerror(phi, phi_ref)
    error_n[i] = tool.rerror(n, n_ref)
    error_max[i] = tool.rerror_max(phi, phi_ref)
    error_max_n[i] = tool.rerror_max(n, n_ref)

tool.plot_convergence("inf_shem361_td_flux", N_particle_list, error, error_max)
tool.plot_convergence("inf_shem361_td_n", N_particle_list, error_n, error_max_n)
