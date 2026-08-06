import numpy as np

import mcdc


def test_lattice_compiles_contained_universes():
    lower_left = mcdc.Universe(name="Lower left")
    lower_right = mcdc.Universe(name="Lower right")
    upper_left = mcdc.Universe(name="Upper left")
    upper_right = mcdc.Universe(name="Upper right")

    lattice = mcdc.Lattice(
        x=(-1.0, 1.0, 2),
        y=(-1.0, 1.0, 2),
        universes=[
            [lower_left, lower_right],
            [upper_left, upper_right],
        ],
    )

    simulation = mcdc.Simulation()
    simulation.set_model([mcdc.Cell(fill=lattice)])
    simulation.compile()

    assert simulation.universes == [
        simulation.root_universe,
        lower_left,
        lower_right,
        upper_left,
        upper_right,
    ]
    np.testing.assert_array_equal(
        lattice.universe_IDs,
        np.array([[[3], [1]], [[4], [2]]]),
    )
