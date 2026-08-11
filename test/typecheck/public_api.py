"""Static checks for representative use of the public MC/DC API."""

from pathlib import Path

import numpy as np

import mcdc


def build_typed_simulation(output: Path) -> mcdc.Simulation:
    neutron_data = mcdc.NeutronMultigroupData(capture=[0.1], scatter=[0.9])
    material = mcdc.Material(name="Material", neutron_multigroup=neutron_data)

    left = mcdc.Surface.PlaneX(x=-1.0, boundary_condition="vacuum")
    right = mcdc.Surface.PlaneX(x=1.0, boundary_condition="vacuum")
    cell = mcdc.Cell(region=+left & -right, fill=material)

    universe = mcdc.Universe(name="Universe", cells=[cell])
    lattice = mcdc.Lattice(x=(-1.0, 2.0, 1), universes=[[universe]])
    root_cell = mcdc.Cell(fill=lattice)

    source = mcdc.Source(position=[0.0, 0.0, 0.0], energy=1.0e6)
    source.move(velocities=[[1.0, 0.0, 0.0]], durations=[1.0])
    left.move(velocities=[[0.5, 0.0, 0.0]], durations=[1.0])

    uniform_mesh = mcdc.MeshUniform(x=(-1.0, 0.1, 20))
    structured_mesh = mcdc.MeshStructured(x=np.linspace(-1.0, 1.0, 21))
    flux = mcdc.Tally(mesh=uniform_mesh, scores=["flux"])
    density = mcdc.Tally(mesh=structured_mesh, scores=["density"])

    simulation = mcdc.Simulation(name="Typed simulation")
    simulation.set_model([root_cell])
    simulation.set_sources([source])
    simulation.set_tallies([flux, density])

    simulation.settings.N_particle = 1_000
    simulation.settings.set_time_census([1.0, 2.0])
    simulation.settings.set_transported_particles(["neutron"])
    simulation.technique.implicit_capture()
    simulation.technique.weight_windows(
        np.ones((1, 20, 1, 1, 3)),
        mesh=uniform_mesh,
        energy=np.array([0.0, 20.0e6]),
    )

    simulation.visualize_model(
        vis_plane="xy",
        x=[-1.0, 1.0],
        y=[-1.0, 1.0],
        z=0.0,
        pixels=[100, 100],
        colors={material: "red"},
        time=[0.0],
        save_as=output,
    )
    return simulation
