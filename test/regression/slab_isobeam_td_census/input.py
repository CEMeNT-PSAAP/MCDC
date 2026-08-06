import numpy as np

import mcdc

# Create MC/DC simulation
simulation = mcdc.Simulation("Slab isotropic beam time-dependent census")

# ======================================================================================
# Set model
# ======================================================================================
# Finite homogeneous pure-absorbing slab

# Set materials
m = mcdc.Material.multigroup(capture=np.array([1.0]))

# Set surfaces
s1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")

# Set cells
cell = mcdc.Cell(region=+s1 & -s2, fill=m)
simulation.set_model([cell])

# ======================================================================================
# Set source
# ======================================================================================
# Isotropic beam from left-end

source = mcdc.Source(
    position=(1e-10, 0.0, 0.0),
    white_direction=(1.0, 0.0, 0.0),
    energy_group=0,
    time=[0.0, 5.0],
)
simulation.set_sources([source])

# ======================================================================================
# Set tallies, settings, techniques, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshUniform(x=(0.0, 0.1, 50))
tally = mcdc.Tally(
    mesh=mesh,
    scores=["flux"],
    time=np.linspace(0.0, 5.0, 51),
)
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 100
simulation.settings.N_batch = 2
simulation.settings.source_bank_buffer_ratio = 5.0
simulation.settings.set_time_census(np.linspace(0.0, 5.0, 6)[1:])

# Techniques
simulation.technique.population_control()

# Run
simulation.run()
