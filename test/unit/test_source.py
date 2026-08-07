import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_
from mcdc.transport.source import source_particle


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


@pytest.mark.parametrize("group", [3, np.int64(3)])
def test_scalar_group(group):
    source = mcdc.Source(group=group)

    assert source.mono_group
    assert source.group == 3


@pytest.mark.parametrize(
    "group",
    [
        [[1, 3], [0.25, 0.75]],
        ([1, 3], [0.25, 0.75]),
        np.array([[1, 3], [0.25, 0.75]]),
    ],
)
def test_group_distribution(group):
    source = mcdc.Source(group=group)

    assert not source.mono_group
    np.testing.assert_array_equal(source.group_pmf.value, [1, 3])


def test_group_distribution_rejects_noninteger_values(capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(group=[[1.5, 3.0], [0.25, 0.75]])

    assert "Group distribution values must be integers" in capsys.readouterr().out


def test_energy_and_group_are_stored_independently():
    source = mcdc.Source(energy=10_000.0, group=3)

    assert source.energy == 10_000.0
    assert source.group == 3


def test_transport_source_sets_energy_and_group_independently(prepare_simulation):
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        energy=10_000.0,
        group=3,
    )
    simulation_container, data = prepare_simulation(sources=[source])
    particle_container = np.zeros(1, dtype=type_.particle)

    source_particle(
        particle_container,
        np.uint64(1),
        simulation_container[0],
        data,
    )

    assert particle_container[0]["E"] == 10_000.0
    assert particle_container[0]["group"] == 3


def test_transport_source_samples_energy_and_group_independently(prepare_simulation):
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        energy=([9_999.0, 10_001.0], [0.5, 0.5]),
        group=([3], [1.0]),
    )
    simulation_container, data = prepare_simulation(sources=[source])
    particle_container = np.zeros(1, dtype=type_.particle)

    source_particle(
        particle_container,
        np.uint64(1),
        simulation_container[0],
        data,
    )

    assert 9_999.0 <= particle_container[0]["E"] <= 10_001.0
    assert particle_container[0]["group"] == 3


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
            {"group": [1, 2, 3]},
            "Group distribution must have shape (2, N)",
        ),
        ({"time": [1.0, 2.0, 3.0]}, "Source time interval must have shape (2,)"),
    ],
)
def test_invalid_distribution_shape(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(**kwargs)

    assert expected_message in capsys.readouterr().out
