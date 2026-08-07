import numpy as np

import mcdc

simulation = mcdc.Simulation("One-group slab shielding")

# Materials
source_region_material = mcdc.Material.multigroup(
    capture=np.array([0.1]),
    scatter=np.array([[0.9]]),
)
shield_material = mcdc.Material.multigroup(
    capture=np.array([0.7]),
    scatter=np.array([[0.3]]),
)

# Geometry
left = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
interface = mcdc.Surface.PlaneZ(z=2.0)
right = mcdc.Surface.PlaneZ(z=6.0, boundary_condition="vacuum")

source_cell = mcdc.Cell(
    region=+left & -interface,
    fill=source_region_material,
)
shield_cell = mcdc.Cell(
    region=+interface & -right,
    fill=shield_material,
)
simulation.set_model([source_cell, shield_cell])

# Source
source = mcdc.Source(
    z=[0.0, 2.0],
    isotropic=True,
    energy=0,
)
simulation.set_sources([source])

# Tally
mesh = mcdc.MeshStructured(z=np.linspace(0.0, 6.0, 61))
flux_tally = mcdc.Tally(
    name="slab_flux",
    mesh=mesh,
    scores=["flux"],
)
simulation.set_tallies([flux_tally])

# Settings
simulation.settings.N_particle = 1_000
simulation.settings.N_batch = 10
simulation.settings.output_name = "slab_shielding"

# Visualize model
"""
simulation.visualize_model(
    vis_plane="xz",
    x=[-1.0, 1.0],
    y=0.0,
    z=[0.0, 6.0],
    pixels=(100, 300),
    colors=None,
    time=[0.0],
    save_as="slab_shielding_geometry",
)
"""

# Run simulation
simulation.run()
