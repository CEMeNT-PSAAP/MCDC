import numpy as np
import pytest

import mcdc
from mcdc.constant import (
    NEUTRON_MULTIGROUP_ENERGY_MIDPOINT,
    NEUTRON_MULTIGROUP_ENERGY_MIDPOINT_LOG,
    NEUTRON_MULTIGROUP_ENERGY_UNIFORM,
    NEUTRON_MULTIGROUP_ENERGY_UNIFORM_LOG,
)
from mcdc.mcdc_get import neutron_multigroup_data as get_neutron_multigroup_data
from mcdc.object_.base import MCDCObject, MCDCPolymorphic
from mcdc.object_.transport_model_data import NeutronMultigroupData


def test_neutron_multigroup_is_public():
    assert mcdc.NeutronMultigroupData is NeutronMultigroupData
    assert not hasattr(mcdc, "NeutronMultigroup")
    assert not hasattr(mcdc, "MGXS")
    assert issubclass(NeutronMultigroupData, MCDCObject)
    assert not issubclass(NeutronMultigroupData, MCDCPolymorphic)
    assert NeutronMultigroupData.label == "neutron_multigroup_data"


def test_zero_group_placeholder():
    neutron_multigroup = NeutronMultigroupData()

    assert neutron_multigroup.G == 0
    assert neutron_multigroup.J == 0
    assert not neutron_multigroup.fissionable
    assert (
        neutron_multigroup.energy_representation == NEUTRON_MULTIGROUP_ENERGY_MIDPOINT
    )
    assert neutron_multigroup._uses_default_energy_grid
    assert "_uses_default_energy_grid" in neutron_multigroup.non_numba

    np.testing.assert_array_equal(neutron_multigroup.energy_grid, [-0.5])
    np.testing.assert_array_equal(neutron_multigroup.speed, [])
    np.testing.assert_array_equal(neutron_multigroup.decay_rate, [])
    np.testing.assert_array_equal(neutron_multigroup.capture, [])
    np.testing.assert_array_equal(neutron_multigroup.scatter, [])
    np.testing.assert_array_equal(neutron_multigroup.fission, [])
    np.testing.assert_array_equal(neutron_multigroup.total, [])
    np.testing.assert_array_equal(neutron_multigroup.nu_s, [])
    np.testing.assert_array_equal(neutron_multigroup.nu_p, [])
    np.testing.assert_array_equal(neutron_multigroup.nu_d, np.empty((0, 0)))
    np.testing.assert_array_equal(neutron_multigroup.nu_d_total, [])
    np.testing.assert_array_equal(neutron_multigroup.nu_f, [])
    np.testing.assert_array_equal(neutron_multigroup.chi_s, np.empty((0, 0)))
    np.testing.assert_array_equal(neutron_multigroup.chi_p, np.empty((0, 0)))
    np.testing.assert_array_equal(neutron_multigroup.chi_d, np.empty((0, 0)))


def test_capture_only_defaults():
    neutron_multigroup = NeutronMultigroupData(capture=[0.1, 0.2])

    assert neutron_multigroup.G == 2
    assert neutron_multigroup.J == 0
    assert not neutron_multigroup.fissionable

    np.testing.assert_array_equal(neutron_multigroup.energy_grid, [-0.5, 0.5, 1.5])
    np.testing.assert_array_equal(neutron_multigroup.speed, [1.0, 1.0])
    np.testing.assert_array_equal(neutron_multigroup.capture, [0.1, 0.2])
    np.testing.assert_array_equal(neutron_multigroup.scatter, [0.0, 0.0])
    np.testing.assert_array_equal(neutron_multigroup.fission, [0.0, 0.0])
    np.testing.assert_array_equal(neutron_multigroup.total, [0.1, 0.2])
    np.testing.assert_array_equal(neutron_multigroup.nu_s, [1.0, 1.0])
    np.testing.assert_array_equal(neutron_multigroup.nu_f, [0.0, 0.0])
    assert neutron_multigroup.capture.dtype == np.float64


def test_documented_two_group_example():
    neutron_multigroup = NeutronMultigroupData(
        capture=[0.1, 0.2],
        scatter=[
            [1.0, 2.0],
            [3.0, 0.0],
        ],
        nu_s=[1.1, 1.2],
        energy_grid=[1.0e-5, 1.0, 20.0e6],
    )

    assert neutron_multigroup.G == 2
    np.testing.assert_array_equal(neutron_multigroup.capture, [0.1, 0.2])
    np.testing.assert_array_equal(neutron_multigroup.scatter, [4.0, 2.0])
    np.testing.assert_allclose(
        neutron_multigroup.chi_s,
        [
            [0.25, 0.75],
            [1.0, 0.0],
        ],
    )
    np.testing.assert_array_equal(neutron_multigroup.total, [4.1, 2.2])
    np.testing.assert_array_equal(neutron_multigroup.nu_s, [1.1, 1.2])
    np.testing.assert_array_equal(neutron_multigroup.energy_grid, [1.0e-5, 1.0, 20.0e6])


def test_prompt_fission_spectrum_is_shared_and_normalized():
    neutron_multigroup = NeutronMultigroupData(
        fission=[0.2, 0.3],
        nu_p=[2.4, 2.5],
        chi_p=[1.0, 3.0],
    )

    assert neutron_multigroup.fissionable
    np.testing.assert_array_equal(neutron_multigroup.nu_f, [2.4, 2.5])
    np.testing.assert_allclose(
        neutron_multigroup.chi_p,
        [
            [0.25, 0.75],
            [0.25, 0.75],
        ],
    )


def test_documented_multiple_precursor_group_example():
    neutron_multigroup = NeutronMultigroupData(
        fission=[0.2, 0.3],
        nu_d=[
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        chi_d=[
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        decay_rate=[0.01, 0.02],
    )

    assert neutron_multigroup.J == 2
    np.testing.assert_array_equal(
        neutron_multigroup.nu_d,
        [
            [0.1, 0.3],
            [0.2, 0.4],
        ],
    )
    np.testing.assert_allclose(neutron_multigroup.nu_d_total, [0.4, 0.6])
    np.testing.assert_allclose(neutron_multigroup.nu_f, [0.4, 0.6])
    np.testing.assert_allclose(
        neutron_multigroup.chi_d,
        [
            [0.25, 0.75],
            [0.75, 0.25],
        ],
    )
    np.testing.assert_array_equal(neutron_multigroup.decay_rate, [0.01, 0.02])


def test_one_group_fission_spectra_default_to_one():
    neutron_multigroup = NeutronMultigroupData(
        fission=[0.2],
        nu_p=[2.4],
        nu_d=[[0.1], [0.2]],
    )

    np.testing.assert_array_equal(neutron_multigroup.chi_p, [[1.0]])
    np.testing.assert_array_equal(neutron_multigroup.chi_d, [[1.0], [1.0]])
    np.testing.assert_array_equal(neutron_multigroup.nu_f, [2.7])
    np.testing.assert_array_equal(neutron_multigroup.decay_rate, [np.inf, np.inf])


def test_standalone_mg_registration_and_packing(prepare_simulation):
    neutron_multigroup = NeutronMultigroupData(
        capture=[0.1, 0.2],
        scatter=[[1.0, 2.0], [3.0, 0.0]],
        energy_grid=[1.0e-5, 1.0, 20.0e6],
    )

    simulation_container, data = prepare_simulation(
        cells=[mcdc.Cell()], objects=[neutron_multigroup]
    )
    simulation = simulation_container[0]
    reserved = simulation["neutron_multigroup_data"][0]
    packed = simulation["neutron_multigroup_data"][1]

    assert simulation["N_neutron_multigroup_data"] == 2
    assert reserved["ID"] == 0
    assert reserved["G"] == 0
    assert neutron_multigroup.ID == 1
    assert packed["ID"] == 1
    assert packed["G"] == 2

    np.testing.assert_array_equal(
        get_neutron_multigroup_data.energy_grid_all(reserved, data), [-0.5]
    )
    np.testing.assert_array_equal(
        get_neutron_multigroup_data.energy_grid_all(packed, data),
        [1.0e-5, 1.0, 20.0e6],
    )
    np.testing.assert_array_equal(
        get_neutron_multigroup_data.capture_all(packed, data), [0.1, 0.2]
    )
    np.testing.assert_array_equal(
        get_neutron_multigroup_data.scatter_all(packed, data), [4.0, 2.0]
    )
    np.testing.assert_allclose(
        get_neutron_multigroup_data.chi_s_vector(0, packed, data),
        [0.25, 0.75],
    )


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("midpoint", NEUTRON_MULTIGROUP_ENERGY_MIDPOINT),
        ("log_midpoint", NEUTRON_MULTIGROUP_ENERGY_MIDPOINT_LOG),
        ("uniform", NEUTRON_MULTIGROUP_ENERGY_UNIFORM),
        ("log_uniform", NEUTRON_MULTIGROUP_ENERGY_UNIFORM_LOG),
        (NEUTRON_MULTIGROUP_ENERGY_MIDPOINT, NEUTRON_MULTIGROUP_ENERGY_MIDPOINT),
        (
            np.int64(NEUTRON_MULTIGROUP_ENERGY_UNIFORM_LOG),
            NEUTRON_MULTIGROUP_ENERGY_UNIFORM_LOG,
        ),
    ],
)
def test_energy_grid_and_representation(policy, expected):
    neutron_multigroup = NeutronMultigroupData(
        capture=[0.1, 0.2],
        energy_grid=[1.0e-5, 1.0, 20.0e6],
        energy_representation=policy,
    )

    assert neutron_multigroup.energy_representation == expected
    assert not neutron_multigroup._uses_default_energy_grid
    np.testing.assert_array_equal(neutron_multigroup.energy_grid, [1.0e-5, 1.0, 20.0e6])


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        (
            {"capture": [0.1, 0.2], "energy_grid": [1.0, 2.0]},
            "energy_grid must have shape (3,)",
        ),
        (
            {"capture": [0.1, 0.2], "energy_grid": [1.0, 2.0, 2.0]},
            "energy grid must be strictly increasing",
        ),
        (
            {
                "capture": [0.1],
                "energy_grid": [0.0, 1.0],
                "energy_representation": "log_midpoint",
            },
            "logarithmic energy representation requires positive",
        ),
        (
            {"capture": [0.1], "energy_representation": "average"},
            "Unknown NeutronMultigroupData energy representation",
        ),
        (
            {"capture": [0.1], "energy_representation": "uniform"},
            "requires an explicit energy_grid",
        ),
        (
            {"fission": [0.1]},
            "NeutronMultigroupData fission data requires nu_p or nu_d",
        ),
        (
            {"nu_p": [2.4]},
            "nu_p must have shape (0,)",
        ),
        (
            {"fission": [0.1, 0.2], "nu_p": [2.4, 2.5]},
            "NeutronMultigroupData with nu_p and G > 1 requires chi_p",
        ),
        (
            {"fission": [0.1, 0.2], "nu_d": [[0.1, 0.2]]},
            "NeutronMultigroupData with nu_d and G > 1 requires chi_d",
        ),
        (
            {"capture": [-0.1]},
            "capture entries must be finite and nonnegative",
        ),
        (
            {"capture": [0.1], "speed": [0.0]},
            "speed entries must be finite and positive",
        ),
        (
            {
                "fission": [0.1, 0.2],
                "nu_p": [2.4, 2.5],
                "chi_p": [0.0, 0.0],
            },
            "chi_p spectrum 0 must have positive mass",
        ),
    ],
)
def test_invalid_inputs_are_rejected(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        NeutronMultigroupData(**kwargs)

    assert expected_message in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        (
            {"capture": [0.1, 0.2], "fission": [0.3], "nu_p": [2.4, 2.5]},
            "fission must have shape (2,)",
        ),
        (
            {"scatter": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
            "scatter must have shape (2, 2)",
        ),
        (
            {"capture": [0.1, 0.2], "speed": [1.0]},
            "speed must have shape (2,)",
        ),
        (
            {
                "fission": [0.1, 0.2],
                "nu_p": [2.4],
                "chi_p": [0.5, 0.5],
            },
            "nu_p must have shape (2,)",
        ),
        (
            {"fission": [0.1, 0.2], "nu_d": [[0.1]]},
            "nu_d must have shape (J, G) with G = 2",
        ),
    ],
)
def test_inconsistent_shapes_are_rejected(kwargs, expected_message, capsys):
    with pytest.raises(SystemExit):
        NeutronMultigroupData(**kwargs)

    assert expected_message in capsys.readouterr().out
