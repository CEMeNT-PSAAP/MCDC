import numpy as np
import os

import mcdc

os.environ["MCDC_LIB"] = "../mcdc-regression_test_data/"

simulation = mcdc.Simulation("Hybrid multigroup regression")

# Native composition combined with multigroup data selects hybrid transport
material = mcdc.Material(
    nuclide_composition={"H1": 5.0e-2},
    neutron_multigroup=mcdc.NeutronMultigroupData(
        capture=np.array([0.2]),
        scatter=np.array([[0.8]]),
        speed=np.array([1.383e6]),
        energy_grid=np.array([0.1, 10.0]),
        energy_representation="midpoint",
    ),
)

boundary = mcdc.Surface.Sphere(radius=5.0, boundary_condition="vacuum")
cell = mcdc.Cell(region=-boundary, fill=material)
simulation.set_model([cell])

multigroup_source = mcdc.Source(
    position=[0.0, 0.0, 1.0],
    isotropic=True,
    energy=1.0,
    probability=0.5,
)
native_source = mcdc.Source(
    position=[0.0, 0.0, -1.0],
    isotropic=True,
    energy=1.0e6,
    probability=0.5,
)
simulation.set_sources([multigroup_source, native_source])

mesh = mcdc.MeshStructured(z=np.array([-5.0, 0.0, 5.0]))
tally = mcdc.Tally(
    name="hybrid_flux",
    mesh=mesh,
    particle_type="neutron",
    energy=np.array([0.0, 0.1, 10.0, 1.0e5, 20.0e6]),
    scores=["flux", "collision"],
)
simulation.set_tallies([tally])

simulation.settings.N_particle = 40
simulation.settings.N_batch = 2

simulation.run()
