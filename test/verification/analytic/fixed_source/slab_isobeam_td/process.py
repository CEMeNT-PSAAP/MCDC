from reference import reference
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
with h5py.File("output_%i.h5" % int(N_particle_list[0]), "r") as f:
    x = f["tally/grid/x"][:]
    t = f["tally/grid/t"][:]
phi_t_ref = reference(x, t)

# Error containers
error_t = np.zeros(len(N_particle_list))
error_max_t = np.zeros(len(N_particle_list))

# Calculate error
for i, N_particle in enumerate(N_particle_list):
    # Get results
    with h5py.File("output_%i.h5" % int(N_particle), "r") as f:
        x = f["tally/grid/x"][:]
        t = f["tally/grid/t"][:]
        K = len(t) - 1
        dx = x[1:] - x[:-1]

        phi_t = f["tally/flux-t/mean"][1:]

    # Normalize
    for k in range(K):
        phi_t[k] *= 0.5 / dx

    # Get error
    error_t[i] = tool.error(phi_t, phi_t_ref)
    error_max_t[i] = tool.error_max(phi_t, phi_t_ref)

# Plot
tool.plot_convergence("slab_isoBeam_td_flux_t", N_particle_list, error_t, error_max_t)
