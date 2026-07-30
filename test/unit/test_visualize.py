import matplotlib
import numpy as np

import mcdc

matplotlib.use("Agg")


def test_visualize_model(tmp_path):
    material = mcdc.MaterialMG(capture=np.array([1.0]))
    left = mcdc.Surface.PlaneZ(z=0.0, boundary_condition="vacuum")
    right = mcdc.Surface.PlaneZ(z=1.0, boundary_condition="vacuum")
    cell = mcdc.Cell(region=+left & -right, fill=material)

    simulation = mcdc.Simulation()
    simulation.set_model([cell])

    output = str(tmp_path / "model")
    simulation.visualize_model(
        vis_plane="xz",
        x=[-0.5, 0.5],
        y=0.0,
        z=[0.0, 1.0],
        pixels=(2, 2),
        colors=None,
        time=[0.0],
        save_as=output,
    )

    assert (tmp_path / "model.png").is_file()
