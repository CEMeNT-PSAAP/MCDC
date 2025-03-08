import numpy as np

import mcdc, mpi4py, h5py

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1
N_history = 50
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
    N_batch = 5,
)
mcdc.time_census(np.linspace(0.0, 20.0, 21)[1:], tally_frequency=1)
'''
mcdc.weight_window(
    x=np.linspace(-20.5, 20.5, 202),
    method="previous",
    modifications=[["min-center", 1e-3]],
    width=2.5,
    save_ww_data=True,
)
'''
mcdc.population_control()
# Run
mcdc.run()
# Combine the tally output into a single file
mcdc.recombine_tallies()