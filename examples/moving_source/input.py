import numpy as np

import mcdc

simulation = mcdc.Simulation("Moving source")

# ======================================================================================
# Set model
# ======================================================================================

# Set materials
air = mcdc.Material.multigroup(
    capture=np.array([0.002]),
    scatter=np.array([[0.008]]),
    speed=np.array([200000.0]),
)

# Set container cell surfaces
min_x = mcdc.Surface.PlaneX(x=-5.0, boundary_condition="vacuum")
max_x = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")
min_y = mcdc.Surface.PlaneY(y=-5.0, boundary_condition="vacuum")
max_y = mcdc.Surface.PlaneY(y=5.0, boundary_condition="vacuum")
min_z = mcdc.Surface.PlaneZ(z=-10.0, boundary_condition="vacuum")
max_z = mcdc.Surface.PlaneZ(z=10.0, boundary_condition="vacuum")

# Make cells
cell = mcdc.Cell(region=+min_x & -max_x & +min_y & -max_y & +min_z & -max_z, fill=air)
simulation.set_model([cell])

# ======================================================================================
# Set source
# ======================================================================================

src = mcdc.Source(
    x=[-4.0, -3.0],
    y=[-0.5, 0.5],
    z=[-0.5, 0.5],
    direction=[1.0, 1.0, 0.0],
    polar_cosine=[-1.0, -0.9],
    energy=0,
    time=[0.0, 10.0],
)
src.move(
    velocities=[
        [1.0, 0.0, 0.0],
        [-0.5, 2.0, 0.0],
        [0.0, -3.0, 0.0],
    ],
    durations=[7.0, 2.0, 1.0],
)
simulation.set_sources([src])

# ======================================================================================
# Set tallies, settings, and run MC/DC
# ======================================================================================

# Tallies
mesh = mcdc.MeshStructured(
    x=np.linspace(-5.0, 5.0, 201),
    y=np.linspace(-5.0, 5.0, 201),
)
tally = mcdc.Tally(mesh=mesh, scores=["flux"], time=np.linspace(0, 10, 46))
simulation.set_tallies([tally])

# Settings
simulation.settings.N_particle = 100000
simulation.settings.N_batch = 2

# Run
simulation.run()
