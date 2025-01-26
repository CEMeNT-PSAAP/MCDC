import numpy as np
from pprint import pprint
import mcdc, mpi4py, h5py

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1

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

mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(-20.5, 20.5, 202),
    t=np.linspace(0.0, 20.0, 21),
)

# Setting
mcdc.setting(N_particle=50, census_bank_buff=5, source_bank_buff=5, N_batch=5)
mcdc.time_census(np.linspace(0.0, 20.0, 5)[1:-1], tally_frequency=5)
mcdc.population_control()

# Run
mcdc.run()

# Combine the tally output into a single file
if mpi4py.MPI.COMM_WORLD.Get_rank() == 0:
    phi = np.zeros((20, 201))
    phi_sd = np.zeros((20, 201))
    N_census = 3
    N_batch = 5
    for i_census in range(N_census):
        for i_batch in range(N_batch):
            with h5py.File(
                "output-batch_%i-census_%i.h5" % (i_batch, i_census), "r"
            ) as f:
                phi_score = f["tallies/mesh_tally_0/flux/score"][:]
                phi[5 * i_census : 5 * i_census + 5, :] += phi_score
                phi_sd[5 * i_census : 5 * i_census + 5, :] += phi_score * phi_score
    phi /= N_batch
    phi_sd = np.sqrt((phi_sd / N_batch - np.square(phi)) / (N_batch - 1))

# Write the results
with h5py.File("output.h5", "a") as f:
    f.create_dataset("tallies/mesh_tally_0/flux/mean", data=phi)
    f.create_dataset("tallies/mesh_tally_0/flux/sdev", data=phi_sd)
