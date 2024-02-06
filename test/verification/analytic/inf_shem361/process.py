from reference import reference
import numpy as np
import h5py
import sys

sys.path.append("../")
import tool


# Cases run
N_min = int(sys.argv[1])
N_max = int(sys.argv[2])
N_particle_list = np.logspace(N_min, N_max, (N_max - N_min) * 2 + 1)

# Reference solution
phi_ref = reference()

# Error container
error = np.zeros(len(N_particle_list))
error_max = np.zeros(len(N_particle_list))

# Calculate error
for k, N_particle in enumerate(N_particle_list):
    # Get results
    with h5py.File("output_%i.h5" % int(N_particle), "r") as f:
        phi = f["tally/flux/mean"][:]

    error[k] = tool.rerror(phi, phi_ref)
    error_max[k] = tool.rerror_max(phi, phi_ref)

# Plot
tool.plot_convergence("inf_shem361_flux", N_particle_list, error, error_max)
