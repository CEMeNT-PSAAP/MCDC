import numpy as np
import mcdc

# Create MC/DC simulation
simulation = mcdc.Simulation("Slab absorbium")

# ======================================================================================
# Set model
# ======================================================================================
# Three slab layers with different purely-absorbing materials

# Set materials
m1 = mcdc.Material.multigroup(capture=np.array([1.0]))
m2 = mcdc.Material.multigroup(capture=np.array([1.5]))
m3 = mcdc.Material.multigroup(capture=np.array([2.0]))

# Set surfaces
s1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneZ(z=2.0)
s3 = mcdc.Surface.PlaneZ(z=4.0)
s4 = mcdc.Surface.PlaneZ(z=6.0, boundary_condition="vacuum")

# Set cells
cell_1 = mcdc.Cell(region=+s1 & -s2, fill=m2)
cell_2 = mcdc.Cell(region=+s2 & -s3, fill=m3)
cell_3 = mcdc.Cell(region=+s3 & -s4, fill=m1)
simulation.set_model([cell_1, cell_2, cell_3])

# ======================================================================================
# Set source
# ======================================================================================
# Uniform isotropic source throughout the domain

source = mcdc.Source(z=[0.0, 6.0], isotropic=True, group=0)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, and run mcdc
# ======================================================================================

# Tallies
surface_tally = mcdc.Tally(surface=s4, scores=["current-net"])
mesh = mcdc.MeshStructured(z=np.linspace(0.0, 6.0, 61))
mesh_tally = mcdc.Tally(
    mesh=mesh,
    mu=np.linspace(-1.0, 1.0, 32 + 1),
    scores=["flux", "collision"],
)
simulation.set_tallies([surface_tally, mesh_tally])

# Settings
simulation.settings.N_particle = 100
simulation.settings.N_batch = 2

# Run
simulation.run()
