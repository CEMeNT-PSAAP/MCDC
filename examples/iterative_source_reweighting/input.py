import numpy as np

import mcdc

simulation = mcdc.Simulation("Iterative source reweighting")

# Homogeneous one-group slab
material = mcdc.Material.multigroup(
    capture=np.array([0.2]),
    scatter=np.array([[0.8]]),
)
left_boundary = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
right_boundary = mcdc.Surface.PlaneZ(z=10.0, boundary_condition="vacuum")
slab = mcdc.Cell(
    region=+left_boundary & -right_boundary,
    fill=material,
)
simulation.set_model([slab])

# Symmetric source regions
source_left = mcdc.Source(
    name="Left source",
    z=[1.0, 2.0],
    isotropic=True,
    group=0,
)
source_right = mcdc.Source(
    name="Right source",
    z=[8.0, 9.0],
    isotropic=True,
    group=0,
)
simulation.set_sources([source_left, source_right])

# One tally is reused by every iteration
mesh = mcdc.MeshStructured(z=np.linspace(0.0, 10.0, 51))
flux_tally = mcdc.Tally(
    name="source_mix_flux",
    mesh=mesh,
    scores=["flux"],
)
simulation.set_tallies([flux_tally])

simulation.settings.N_particle = 20_000
simulation.settings.N_batch = 10

# Partially update the same owned model and compile a fresh snapshot each time.
source_mixes = (
    ("left_20", 0.2),
    ("left_50", 0.5),
    ("left_80", 0.8),
)

for case_name, left_fraction in source_mixes:
    source_left.probability = left_fraction
    source_right.probability = 1.0 - left_fraction
    simulation.settings.output_name = f"source_mix_{case_name}"

    simulation.compile()
    simulation.run()
