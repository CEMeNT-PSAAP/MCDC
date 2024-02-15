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
with h5py.File("output_%i.h5" % int(N_particle_list[0]), "r") as f:
    z = f["tally/grid/z"][:]
    mu = f["tally/grid/mu"][:]
phi_ref, J_ref, psi_ref = reference(z, mu)

# Error containers
error = np.zeros(len(N_particle_list))
error_J = np.zeros(len(N_particle_list))
error_psi = np.zeros(len(N_particle_list))

error_max = np.zeros(len(N_particle_list))
error_max_J = np.zeros(len(N_particle_list))
error_max_psi = np.zeros(len(N_particle_list))

# Calculate error
for k, N_particle in enumerate(N_particle_list):
    # Get results
    with h5py.File("output_%i.h5" % int(N_particle), "r") as f:
        z = f["tally/grid/z"][:]
        dz = z[1:] - z[:-1]
        mu = f["tally/grid/mu"][:]
        dmu = mu[1:] - mu[:-1]
        I = len(z) - 1
        N = len(mu) - 1

        psi = f["tally/flux/mean"][:]
        J = f["tally/current/mean"][:, 2]

    # Scalar flux
    phi = np.zeros(I)
    for i in range(I):
        phi[i] += np.sum(psi[i, :])

    psi_norm = np.zeros(psi.shape)
    # Normalize
    phi /= dz
    J /= dz
    for n in range(N):
        psi[:, n] = psi[:, n] / dz / dmu[n]

    # Get error
    error[k] = tool.rerror(phi, phi_ref)
    error_J[k] = tool.rerror(J, J_ref)
    error_psi[k] = tool.rerror(psi, psi_ref)

    error_max[k] = tool.rerror_max(phi, phi_ref)
    error_max_J[k] = tool.rerror_max(J, J_ref)
    error_max_psi[k] = tool.rerror_max(psi, psi_ref)


# Plot
tool.plot_convergence("slab_absorbium_flux", N_particle_list, error, error_max)
tool.plot_convergence("slab_absorbium_current", N_particle_list, error_J, error_max_J)
tool.plot_convergence(
    "slab_absorbium_angular_flux", N_particle_list, error_psi, error_max_psi
)
