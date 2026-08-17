from types import SimpleNamespace

import h5py
import numpy as np

from mcdc.constant import SCORE_FLUX
from mcdc.output import clear_census_based_tally_files, recombine_tallies


def _write_census_tally(path, values):
    with h5py.File(path, "w") as file:
        tally = file.require_group("tallies/tracklength_tally_0")
        tally.require_group("grid").create_dataset("time", data=[0.0, 1.0])
        score = tally.require_group("flux")
        score.create_dataset("mean", data=values)
        score.create_dataset("sdev", data=np.zeros_like(values))


def test_clear_census_based_tally_files(tmp_path):
    base_name = tmp_path / "output"
    settings = SimpleNamespace(output_name=str(base_name), N_batch=2, N_census=3)
    expected_files = [
        tmp_path / f"output-batch_{idx_batch}-census_{idx_census}.h5"
        for idx_batch in range(settings.N_batch)
        for idx_census in range(settings.N_census)
    ]
    for path in expected_files:
        path.touch()
    unrelated = tmp_path / "output-unrelated.h5"
    unrelated.touch()

    clear_census_based_tally_files(settings)

    assert not any(path.exists() for path in expected_files)
    assert unrelated.exists()


def test_recombine_tallies_zero_fills_censuses_missing_after_extinction(tmp_path):
    base_name = tmp_path / "output"
    settings = SimpleNamespace(
        output_name=str(base_name),
        use_census_based_tally=True,
        census_tally_frequency=1,
        census_time=np.array([1.0, 2.0, np.inf]),
        N_batch=2,
        N_census=3,
    )
    tally = SimpleNamespace(
        name="tracklength_tally_0",
        scores=[SCORE_FLUX],
        bin_shape=[1, 1, 1, 1, 2, 1],
    )
    simulation_python = SimpleNamespace(settings=settings, tallies=[tally])
    simulation = {"mpi_master": True}

    with h5py.File(f"{base_name}.h5", "w"):
        pass

    _write_census_tally(f"{base_name}-batch_0-census_0.h5", np.array([2.0, 4.0]))
    _write_census_tally(f"{base_name}-batch_1-census_0.h5", np.array([4.0, 6.0]))
    _write_census_tally(f"{base_name}-batch_1-census_1.h5", np.array([8.0, 10.0]))

    recombine_tallies(simulation_python, simulation)

    with h5py.File(f"{base_name}.h5", "r") as file:
        tally_path = "tallies/tracklength_tally_0"
        np.testing.assert_array_equal(
            file[f"{tally_path}/grid/time"][:], [0.0, 1.0, 2.0]
        )
        np.testing.assert_array_equal(
            file[f"{tally_path}/flux/mean"][:],
            [[3.0, 5.0], [4.0, 5.0]],
        )
        np.testing.assert_array_equal(
            file[f"{tally_path}/flux/sdev"][:],
            [[1.0, 1.0], [4.0, 5.0]],
        )
