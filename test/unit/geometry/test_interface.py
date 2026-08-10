import numpy as np

import mcdc
import mcdc.numba_types as type_
from mcdc.transport.geometry import locate_particle


def test_locate_particle_without_material_speed(prepare_simulation):
    material = mcdc.Material.multigroup(capture=np.array([1.0]))
    boundary = mcdc.Surface.PlaneX(x=0.0)
    universe = mcdc.Universe(cells=[mcdc.Cell(region=+boundary, fill=material)])
    root_cell = mcdc.Cell(fill=universe, translation=[5.0, 0.0, 0.0])
    simulation_container, data = prepare_simulation(cells=[root_cell])
    simulation = simulation_container[0]

    particle_container = np.zeros(1, dtype=type_.particle)
    particle = particle_container[0]
    particle["x"] = 6.0
    particle["ux"] = 1.0
    particle["cell_ID"] = -1
    particle["material_ID"] = -1
    original_coordinates = tuple(
        particle[field] for field in ("x", "y", "z", "t", "ux", "uy", "uz")
    )

    found = locate_particle(particle_container, simulation, data)

    assert found
    assert particle["cell_ID"] == root_cell.ID
    assert particle["material_ID"] == material.ID
    assert (
        tuple(particle[field] for field in ("x", "y", "z", "t", "ux", "uy", "uz"))
        == original_coordinates
    )
