import math

import numpy as np
import pytest

import mcdc
import mcdc.numba_types as type_
import mcdc.transport.physics.neutron.multigroup as multigroup


def _prepare_multigroup(prepare_simulation, **kwargs):
    material = mcdc.Material.multigroup(
        energy_grid=[1.0, 10.0, 100.0],
        **kwargs,
    )
    simulation_container, data = prepare_simulation(cells=[mcdc.Cell(fill=material)])
    simulation = simulation_container[0]
    packed_material = simulation["materials"][0]
    mgxs = simulation["neutron_multigroup_data"][
        packed_material["neutron_multigroup_ID"]
    ]
    return simulation, mgxs, data


@pytest.mark.parametrize(
    "representation, expected",
    [
        ("midpoint", 55.0),
        ("log_midpoint", math.sqrt(1000.0)),
        ("uniform", None),
        ("log_uniform", None),
    ],
)
def test_group_energy_representation(representation, expected, prepare_simulation):
    simulation, mgxs, data = _prepare_multigroup(
        prepare_simulation,
        capture=[1.0, 1.0],
        energy_representation=representation,
    )
    simulation["technique"]["neutron_multigroup"]["hybrid"] = True

    particle_container = np.zeros(1, dtype=type_.particle)
    particle_container[0]["group"] = 1
    particle_container[0]["rng_seed"] = np.uint64(1)

    energy = multigroup._get_group_energy(particle_container, mgxs, simulation, data)

    if expected is None:
        assert 10.0 <= energy < 100.0
    else:
        assert energy == pytest.approx(expected)


def test_standard_multigroup_does_not_reconstruct_energy(prepare_simulation):
    simulation, mgxs, data = _prepare_multigroup(
        prepare_simulation,
        capture=[1.0, 1.0],
    )

    particle_container = np.zeros(1, dtype=type_.particle)
    particle_container[0]["group"] = 1

    energy = multigroup._get_group_energy(particle_container, mgxs, simulation, data)

    assert energy == 0.0


def test_hybrid_energy_groups_are_left_closed(prepare_simulation):
    simulation, mgxs, data = _prepare_multigroup(
        prepare_simulation,
        capture=[1.0, 1.0],
    )
    simulation["technique"]["neutron_multigroup"]["hybrid"] = True

    particle_container = np.zeros(1, dtype=type_.particle)
    particle = particle_container[0]
    particle["material_ID"] = 0

    particle["E"] = 1.0
    assert multigroup.applicable(particle_container, simulation, data)
    assert multigroup._get_energy_group(particle_container, mgxs, simulation, data) == 0

    particle["E"] = 10.0
    assert multigroup._get_energy_group(particle_container, mgxs, simulation, data) == 1

    particle["E"] = 100.0
    assert not multigroup.applicable(particle_container, simulation, data)


def _make_particle():
    particle_container = np.zeros(1, dtype=type_.particle)
    particle = particle_container[0]
    particle["material_ID"] = 0
    particle["alive"] = True
    particle["E"] = 5.0
    particle["w"] = 1.0
    particle["uz"] = 1.0
    particle["rng_seed"] = np.uint64(1)
    return particle_container


def _assert_fission_products(particle_container, simulation):
    particle = particle_container[0]
    assert particle["alive"]
    assert particle["group"] == 1
    assert particle["E"] == pytest.approx(55.0)

    bank = simulation["bank_active"]
    assert bank["size"][0] == 3
    banked = bank["particle_data"][:3]
    np.testing.assert_array_equal(banked["group"], 1)
    np.testing.assert_allclose(banked["E"], 55.0)


def test_scattering_product_uses_multigroup_data(prepare_simulation):
    simulation, _, data = _prepare_multigroup(
        prepare_simulation,
        scatter=[[0.0, 0.0], [1.0, 1.0]],
    )
    simulation["technique"]["neutron_multigroup"]["hybrid"] = True
    particle_container = _make_particle()

    multigroup.scattering(particle_container, simulation, data)

    particle = particle_container[0]
    assert particle["alive"]
    assert particle["group"] == 1
    assert particle["E"] == pytest.approx(55.0)


def test_prompt_fission_products(prepare_simulation):
    simulation, _, data = _prepare_multigroup(
        prepare_simulation,
        fission=[1.0, 1.0],
        nu_p=[4.0, 4.0],
        chi_p=[0.0, 1.0],
    )
    simulation["technique"]["neutron_multigroup"]["hybrid"] = True
    particle_container = _make_particle()

    multigroup.fission(particle_container, simulation, data)

    _assert_fission_products(particle_container, simulation)
    assert particle_container[0]["t"] == 0.0


def test_delayed_fission_products(prepare_simulation):
    simulation, _, data = _prepare_multigroup(
        prepare_simulation,
        fission=[1.0, 1.0],
        nu_d=[[4.0, 4.0]],
        chi_d=[[0.0], [1.0]],
        decay_rate=[1.0],
    )
    simulation["technique"]["neutron_multigroup"]["hybrid"] = True
    particle_container = _make_particle()

    multigroup.fission(particle_container, simulation, data)

    _assert_fission_products(particle_container, simulation)
    assert particle_container[0]["t"] > 0.0
    assert np.all(simulation["bank_active"]["particle_data"][:3]["t"] > 0.0)
