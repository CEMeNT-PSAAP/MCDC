import numpy as np
import mcdc

# Create MC/DC simulation
simulation = mcdc.Simulation("AZURV1 census tally")

# ======================================================================================
# Set model
# ======================================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1

# Set materials
m = mcdc.Material.multigroup(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2.3]),
)

# Set surfaces
s1 = mcdc.Surface.PlaneX(x=-1e10, boundary_condition="reflective")
s2 = mcdc.Surface.PlaneX(x=1e10, boundary_condition="reflective")

# Set cells
cell = mcdc.Cell(region=+s1 & -s2, fill=m)
simulation.set_model([cell])

# ======================================================================================
# Set source
# ======================================================================================
# Isotropic pulse at x=t=0

source = mcdc.Source(
    position=[0.0, 0.0, 0.0],
    isotropic=True,
    group=0,
    time=0.0,
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, techniques, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshStructured(x=np.linspace(-20.5, 20.5, 202))
tally = mcdc.Tally(mesh=mesh, scores=["flux"], time=np.linspace(0.0, 20.0, 21))
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 50
simulation.settings.N_batch = 2
simulation.settings.census_bank_buffer_ratio = 5.0
simulation.settings.source_bank_buffer_ratio = 5.0
simulation.settings.set_time_census(np.linspace(0.0, 20.0, 21)[1:], tally_frequency=5)

# Techniques
simulation.technique.population_control()

# Run
simulation.run()
