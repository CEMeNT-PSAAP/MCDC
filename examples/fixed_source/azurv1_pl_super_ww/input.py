import numpy as np

import mcdc, mpi4py, h5py

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1
N_history = 1e2
# Set materials
m = mcdc.material(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2.3]),
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

# Set cells
mcdc.cell(+s1 & -s2, m)

# =============================================================================
# Set source
# =============================================================================
# Isotropic pulse at x=t=0

mcdc.source(point=[0.0, 0.0, 0.0], isotropic=True, time=[1e-10, 1e-10])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average, cell-edge, and time-edge scalar fluxes
mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(-20.5, 20.5, 202),
    t=np.linspace(0.0, 20.0, 21),
)

# Setting
mcdc.setting(
    N_particle=N_history,
    active_bank_buff=1e4,
    census_bank_buff=1e3,
    source_bank_buff=1e3,
)
mcdc.time_census(np.linspace(0.0, 20.0, 21)[1:], tally_frequency=1)

mcdc.weight_window(
    x=np.linspace(-20.5, 20.5, 202),
    method="previous",
    modifications=[["min-center", 1e-3]],
    width=2.5,
    save_ww_data=True,
)

# Run
mcdc.run()
# Combine the tally output into a single file
if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    phi = np.zeros((20, 201))
    phi_sd = np.zeros((20, 201))
    centers = np.zeros((20, 201))
    N_census = 20
    N_batch = 1
    N_tallies = 1
    for i_census in range(N_census):
        for i_batch in range(N_batch):
            with h5py.File(
                "output-batch_%i-census_%i.h5" % (i_batch, i_census), "r"
            ) as f:
                phi_score = f["tallies/mesh_tally_0/flux/score"][:]
                window_centers = f["weight_window_centers"][:]
                phi[
                    N_tallies * i_census : N_tallies * i_census + N_tallies, :
                ] += phi_score
                phi_sd[N_tallies * i_census : N_tallies * i_census + N_tallies, :] += (
                    phi_score * phi_score
                )
                centers[
                    N_tallies * i_census : N_tallies * i_census + N_tallies, :
                ] += np.squeeze(window_centers)
    phi /= N_batch
    phi_sd = np.sqrt(
        (phi_sd / N_history - np.square(phi / N_history)) / (N_history - 1)
    )
    # Write the results
    with h5py.File("output.h5", "a") as f:
        f.create_dataset("tallies/mesh_tally_0/flux/mean", data=phi)
        f.create_dataset("tallies/mesh_tally_0/flux/sdev", data=phi_sd)
        f.create_dataset("weight_window_centers", data=centers)
