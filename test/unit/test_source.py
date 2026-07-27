import numpy as np
import pytest

import mcdc


@pytest.mark.parametrize(
    "energy",
    [
        10_000,
        10_000.0,
        np.int64(10_000),
        np.float64(10_000.0),
    ],
)
def test_scalar_energy(energy):
    source = mcdc.Source(energy=energy)

    assert source.mono_energetic
    assert source.energy == 10_000.0


@pytest.mark.parametrize(
    "energy",
    [
        [[9_999.0, 10_001.0], [0.5, 0.5]],
        ([9_999.0, 10_001.0], [0.5, 0.5]),
        np.array([[9_999.0, 10_001.0], [0.5, 0.5]]),
    ],
)
def test_energy_distribution(energy):
    source = mcdc.Source(energy=energy)

    assert not source.mono_energetic
    np.testing.assert_array_equal(source.energy_pdf.pdf.x, [9_999.0, 10_001.0])


@pytest.mark.parametrize("energy_group", [3, np.int64(3)])
def test_scalar_energy_group(energy_group):
    source = mcdc.Source(energy_group=energy_group)

    assert source.mono_energetic
    assert source.energy_group == 3


@pytest.mark.parametrize(
    "energy_group",
    [
        [[1, 3], [0.25, 0.75]],
        ([1, 3], [0.25, 0.75]),
        np.array([[1, 3], [0.25, 0.75]]),
    ],
)
def test_energy_group_distribution(energy_group):
    source = mcdc.Source(energy_group=energy_group)

    assert not source.mono_energetic
    np.testing.assert_array_equal(source.energy_group_pmf.value, [1, 3])


@pytest.mark.parametrize("time", [2, 2.0, np.int64(2), np.float64(2.0)])
def test_scalar_time(time):
    source = mcdc.Source(time=time)

    assert source.discrete_time
    assert source.time == 2.0


@pytest.mark.parametrize(
    "time",
    [
        [1.0, 2.0],
        (1.0, 2.0),
        np.array([1.0, 2.0]),
    ],
)
def test_time_interval(time):
    source = mcdc.Source(time=time)

    assert not source.discrete_time
    np.testing.assert_array_equal(source.time_range, [1.0, 2.0])


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"energy": [1.0, 2.0, 3.0]}, "Energy distribution must have shape (2, N)"),
        (
            {"energy_group": [1, 2, 3]},
            "Energy-group distribution must have shape (2, N)",
        ),
        ({"time": [1.0, 2.0, 3.0]}, "Source time interval must have shape (2,)"),
    ],
)
def test_invalid_distribution_shape(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(**kwargs)

    assert expected_message in capsys.readouterr().out
