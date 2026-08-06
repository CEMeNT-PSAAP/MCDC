import numpy as np
import mcdc

# Create MC/DC simulation
simulation = mcdc.Simulation("Reed slab")

# ======================================================================================
# Set model
# ======================================================================================
# Three slab layers with different materials
# Based on William H. Reed, NSE (1971), 46:2, 309-314, DOI: 10.13182/NSE46-309

# Set materials
m1 = mcdc.MaterialMG(capture=np.array([50.0]))
m2 = mcdc.MaterialMG(capture=np.array([5.0]))
m3 = mcdc.MaterialMG(capture=np.array([0.0]))  # Vacuum
m4 = mcdc.MaterialMG(capture=np.array([0.1]), scatter=np.array([[0.9]]))

# Set surfaces
s1 = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="reflective")
s2 = mcdc.Surface.PlaneZ(z=2.0)
s3 = mcdc.Surface.PlaneZ(z=3.0)
s4 = mcdc.Surface.PlaneZ(z=5.0)
s5 = mcdc.Surface.PlaneZ(z=8.0, boundary_condition="vacuum")

# Set cells
cell_1 = mcdc.Cell(region=+s1 & -s2, fill=m1)
cell_2 = mcdc.Cell(region=+s2 & -s3, fill=m2)
cell_3 = mcdc.Cell(region=+s3 & -s4, fill=m3)
cell_4 = mcdc.Cell(region=+s4 & -s5, fill=m4)
simulation.set_model([cell_1, cell_2, cell_3, cell_4])

# ======================================================================================
# Set source
# ======================================================================================

# Isotropic source in the absorbing medium
source_1 = mcdc.Source(z=[0.0, 2.0], isotropic=True, energy_group=0, probability=50.0)

# Isotropic source in the first half of the outermost medium,
# with 1/100 strength
source_2 = mcdc.Source(z=[5.0, 6.0], isotropic=True, energy_group=0, probability=0.5)
simulation.set_sources([source_1, source_2])

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshStructured(z=np.linspace(0.0, 8.0, 81))
tally = mcdc.Tally(mesh=mesh, scores=["flux"])
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 4000
simulation.settings.N_batch = 2

# Run
simulation.run()
