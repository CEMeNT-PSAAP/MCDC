import numpy as np

import mcdc

# Create MC/DC simulation
simulation = mcdc.Simulation("Infinite SHEM-361 k-eigenvalue")

# =============================================================================
# Set model
# =============================================================================
# The infinite homogenous medium is modeled with reflecting slab

# Load material data
with np.load("SHEM-361.npz") as data:
    SigmaC = data["SigmaC"]  # /cm
    SigmaS = data["SigmaS"]
    SigmaF = data["SigmaF"]
    nu_p = data["nu_p"]
    nu_d = data["nu_d"]
    chi_p = data["chi_p"]
    chi_d = data["chi_d"]
    G = data["G"]

# Set material
m = mcdc.Material.multigroup(
    capture=SigmaC,
    scatter=SigmaS,
    fission=SigmaF,
    nu_p=nu_p,
    chi_p=chi_p,
    nu_d=nu_d,
    chi_d=chi_d,
)

# Set surfaces
s1 = mcdc.Surface.PlaneX(x=-1e10, boundary_condition="reflective")
s2 = mcdc.Surface.PlaneX(x=1e10, boundary_condition="reflective")

# Set cells
c = mcdc.Cell(region=+s1 & -s2, fill=m)
simulation.set_model([c])

# =============================================================================
# Set initial source
# =============================================================================

source = mcdc.Source(
    position=(0.0, 0.0, 0.0), isotropic=True, discrete_energy=np.array([[360], [1.0]])
)
simulation.set_sources([source])

# =============================================================================
# Set tallies, settings, techniques, and run MC/DC
# =============================================================================

# Tallies
tally = mcdc.Tally(scores=["flux"], energy="all")
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 70
simulation.settings.source_bank_buffer_ratio = 2.0
simulation.settings.census_bank_buffer_ratio = 3.0
simulation.settings.set_eigenmode(N_inactive=1, N_active=2)

# Techniques
simulation.technique.population_control()

# Run
simulation.run()
