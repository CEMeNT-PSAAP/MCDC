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
    assert not source.discrete_energy
    np.testing.assert_array_equal(source.energy_pdf.pdf.x, [9_999.0, 10_001.0])


@pytest.mark.parametrize(
    "discrete_energy",
    [
        [[9_999.0, 10_001.0], [0.25, 0.75]],
        ([9_999.0, 10_001.0], [0.25, 0.75]),
        np.array([[9_999.0, 10_001.0], [0.25, 0.75]]),
    ],
)
def test_discrete_energy_distribution(discrete_energy):
    source = mcdc.Source(discrete_energy=discrete_energy)

    assert not source.mono_energetic
    assert source.discrete_energy
    np.testing.assert_array_equal(source.energy_pmf.value, [9_999.0, 10_001.0])
    assert "Energy: PMF" in repr(source)


def test_energy_and_discrete_energy_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(
            energy=10_000.0,
            discrete_energy=[[9_999.0, 10_001.0], [0.25, 0.75]],
        )

    assert "Cannot specify both energy and discrete_energy" in capsys.readouterr().out


def test_transport_source_sets_mono_energy(prepare_simulation):
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        energy=10_000.0,
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


def test_transport_source_samples_discrete_energy(prepare_simulation):
    source = mcdc.Source(
        position=[0.0, 0.0, 0.0],
        direction=[0.0, 0.0, 1.0],
        discrete_energy=([10_001.0], [1.0]),
    )
    simulation_container, data = prepare_simulation(sources=[source])
    particle_container = np.zeros(1, dtype=type_.particle)

    source_particle(
        particle_container,
        np.uint64(1),
        simulation_container[0],
        data,
    )

    assert particle_container[0]["E"] == 10_001.0


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
            {"discrete_energy": [1.0, 2.0, 3.0]},
            "Discrete energy distribution must have shape (2, N)",
        ),
        ({"time": [1.0, 2.0, 3.0]}, "Source time interval must have shape (2,)"),
    ],
)
def test_invalid_distribution_shape(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        mcdc.Source(**kwargs)

    assert expected_message in capsys.readouterr().out
