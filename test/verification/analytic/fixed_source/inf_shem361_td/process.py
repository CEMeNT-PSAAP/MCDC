from plotter import plot_convergence
import numpy as np
import h5py
import sys

sys.path.append("../../util")


N_min = int(sys.argv[1])
N_max = int(sys.argv[2])
N_particle_list = np.logspace(N_min, N_max, (N_max - N_min) * 2 + 1)

# Reference solution
data = np.load("reference.npz")
phi_ref = data["phi"].T
n_ref = data["n"].T

error = []
error_n = []

for N_particle in N_particle_list:
    # Get results
    with h5py.File("output_%i.h5" % int(N_particle), "r") as f:
        phi = f["tally/flux-t/mean"][:]
    phi = phi[:, 1:]
    with np.load("CASMO-70.npz") as data:
        speed = data["v"]

    # Neutron density
    n = np.zeros(K)
    for k in range(K):
        n[k] = np.sum(phi[:, k] / speed)

    error.append(np.linalg.norm((phi - phi_ref) / phi_ref))
    error_n.append(np.linalg.norm((n - n_ref) / n_ref))

plot_convergence("inf_casmo70_td_flux", N_particle_list, error)
plot_convergence("inf_casmo70_td_n", N_particle_list, error_n)
