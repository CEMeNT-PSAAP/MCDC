import numpy as np
import mcdc

# Create MC/DC simulation
simulation = mcdc.Simulation("AZURV1")

# ======================================================================================
# Set simulation model
# ======================================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1

# Materials
m = mcdc.Material.multigroup(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2.3]),
)

# Surfaces
s1 = mcdc.Surface.PlaneX(x=-1e10, boundary_condition="reflective")
s2 = mcdc.Surface.PlaneX(x=1e10, boundary_condition="reflective")

# Cells
cell = mcdc.Cell(region=+s1 & -s2, fill=m)

# Set model
simulation.set_model([cell])

# ======================================================================================
# Set simulation sources
# ======================================================================================
# Isotropic pulse at x=t=0

source = mcdc.Source(
    position=[0.0, 0.0, 0.0],
    isotropic=True,
    energy=0,
    time=0.0,
)
simulation.set_sources([source])

# ======================================================================================
# Set simulation tallies
# ======================================================================================

mesh = mcdc.MeshStructured(x=np.linspace(-20.5, 20.5, 202))
tally = mcdc.Tally(mesh=mesh, scores=["flux"], time=np.linspace(0.0, 20.0, 21))
simulation.set_tallies([tally])

# ======================================================================================
# Simulation settings and run simulation
# ======================================================================================

# Settings
simulation.settings.N_particle = 60
simulation.settings.N_batch = 2

# Run
simulation.run()
