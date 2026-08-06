import numpy as np
import pytest

import mcdc
from mcdc.constant import (
    MULTIGROUP_NEUTRON_ENERGY_MIDPOINT,
    MULTIGROUP_NEUTRON_ENERGY_MIDPOINT_LOG,
    MULTIGROUP_NEUTRON_ENERGY_UNIFORM,
    MULTIGROUP_NEUTRON_ENERGY_UNIFORM_LOG,
)
from mcdc.object_.base import MCDCObject, MCDCPolymorphic
from mcdc.object_.mgxs import MGXS


def test_mgxs_is_public():
    assert mcdc.MGXS is MGXS
    assert issubclass(MGXS, MCDCObject)
    assert not issubclass(MGXS, MCDCPolymorphic)


def test_zero_group_placeholder():
    mgxs = MGXS()

    assert mgxs.G == 0
    assert mgxs.J == 0
    assert not mgxs.fissionable
    assert not mgxs.has_energy_grid
    assert mgxs.energy_representation == MULTIGROUP_NEUTRON_ENERGY_MIDPOINT_LOG

    np.testing.assert_array_equal(mgxs.energy_grid, [0.0])
    np.testing.assert_array_equal(mgxs.speed, [])
    np.testing.assert_array_equal(mgxs.decay_rate, [])
    np.testing.assert_array_equal(mgxs.capture, [])
    np.testing.assert_array_equal(mgxs.scatter, [])
    np.testing.assert_array_equal(mgxs.fission, [])
    np.testing.assert_array_equal(mgxs.total, [])
    np.testing.assert_array_equal(mgxs.nu_s, [])
    np.testing.assert_array_equal(mgxs.nu_p, [])
    np.testing.assert_array_equal(mgxs.nu_d, np.empty((0, 0)))
    np.testing.assert_array_equal(mgxs.nu_d_total, [])
    np.testing.assert_array_equal(mgxs.nu_f, [])
    np.testing.assert_array_equal(mgxs.chi_s, np.empty((0, 0)))
    np.testing.assert_array_equal(mgxs.chi_p, np.empty((0, 0)))
    np.testing.assert_array_equal(mgxs.chi_d, np.empty((0, 0)))


def test_capture_only_defaults():
    mgxs = MGXS(capture=[0.1, 0.2])

    assert mgxs.G == 2
    assert mgxs.J == 0
    assert not mgxs.fissionable
    assert not mgxs.has_energy_grid

    np.testing.assert_array_equal(mgxs.energy_grid, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(mgxs.speed, [1.0, 1.0])
    np.testing.assert_array_equal(mgxs.capture, [0.1, 0.2])
    np.testing.assert_array_equal(mgxs.scatter, [0.0, 0.0])
    np.testing.assert_array_equal(mgxs.fission, [0.0, 0.0])
    np.testing.assert_array_equal(mgxs.total, [0.1, 0.2])
    np.testing.assert_array_equal(mgxs.nu_s, [1.0, 1.0])
    np.testing.assert_array_equal(mgxs.nu_f, [0.0, 0.0])
    assert mgxs.capture.dtype == np.float64


def test_scatter_matrix_is_reoriented_and_normalized():
    mgxs = MGXS(
        scatter=[
            [1.0, 2.0],
            [3.0, 0.0],
        ],
        nu_s=[1.1, 1.2],
    )

    np.testing.assert_array_equal(mgxs.scatter, [4.0, 2.0])
    np.testing.assert_allclose(
        mgxs.chi_s,
        [
            [0.25, 0.75],
            [1.0, 0.0],
        ],
    )
    np.testing.assert_array_equal(mgxs.total, [4.0, 2.0])
    np.testing.assert_array_equal(mgxs.nu_s, [1.1, 1.2])


def test_prompt_fission_spectrum_is_shared_and_normalized():
    mgxs = MGXS(
        fission=[0.2, 0.3],
        nu_p=[2.4, 2.5],
        chi_p=[1.0, 3.0],
    )

    assert mgxs.fissionable
    np.testing.assert_array_equal(mgxs.nu_f, [2.4, 2.5])
    np.testing.assert_allclose(
        mgxs.chi_p,
        [
            [0.25, 0.75],
            [0.25, 0.75],
        ],
    )


def test_delayed_fission_data_is_reoriented_and_normalized():
    mgxs = MGXS(
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

    assert mgxs.J == 2
    np.testing.assert_array_equal(
        mgxs.nu_d,
        [
            [0.1, 0.3],
            [0.2, 0.4],
        ],
    )
    np.testing.assert_allclose(mgxs.nu_d_total, [0.4, 0.6])
    np.testing.assert_allclose(mgxs.nu_f, [0.4, 0.6])
    np.testing.assert_allclose(
        mgxs.chi_d,
        [
            [0.25, 0.75],
            [0.75, 0.25],
        ],
    )
    np.testing.assert_array_equal(mgxs.decay_rate, [0.01, 0.02])


def test_one_group_fission_spectra_default_to_one():
    mgxs = MGXS(
        fission=[0.2],
        nu_p=[2.4],
        nu_d=[[0.1], [0.2]],
    )

    np.testing.assert_array_equal(mgxs.chi_p, [[1.0]])
    np.testing.assert_array_equal(mgxs.chi_d, [[1.0], [1.0]])
    np.testing.assert_array_equal(mgxs.nu_f, [2.7])
    np.testing.assert_array_equal(mgxs.decay_rate, [np.inf, np.inf])


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("midpoint", MULTIGROUP_NEUTRON_ENERGY_MIDPOINT),
        ("log_midpoint", MULTIGROUP_NEUTRON_ENERGY_MIDPOINT_LOG),
        ("uniform", MULTIGROUP_NEUTRON_ENERGY_UNIFORM),
        ("log_uniform", MULTIGROUP_NEUTRON_ENERGY_UNIFORM_LOG),
        (MULTIGROUP_NEUTRON_ENERGY_MIDPOINT, MULTIGROUP_NEUTRON_ENERGY_MIDPOINT),
        (
            np.int64(MULTIGROUP_NEUTRON_ENERGY_UNIFORM_LOG),
            MULTIGROUP_NEUTRON_ENERGY_UNIFORM_LOG,
        ),
    ],
)
def test_energy_grid_and_representation(policy, expected):
    mgxs = MGXS(
        capture=[0.1, 0.2],
        energy_grid=[1.0e-5, 1.0, 20.0e6],
        energy_representation=policy,
    )

    assert mgxs.has_energy_grid
    assert mgxs.energy_representation == expected
    np.testing.assert_array_equal(mgxs.energy_grid, [1.0e-5, 1.0, 20.0e6])


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
            {"capture": [0.1], "energy_grid": [0.0, 1.0]},
            "logarithmic energy representation requires positive",
        ),
        (
            {"capture": [0.1], "energy_representation": "average"},
            "Unknown MGXS energy representation",
        ),
        (
            {"fission": [0.1]},
            "Fission MGXS requires nu_p or nu_d",
        ),
        (
            {"nu_p": [2.4]},
            "nu_p must have shape (0,)",
        ),
        (
            {"fission": [0.1, 0.2], "nu_p": [2.4, 2.5]},
            "MGXS with nu_p and G > 1 requires chi_p",
        ),
        (
            {"fission": [0.1, 0.2], "nu_d": [[0.1, 0.2]]},
            "MGXS with nu_d and G > 1 requires chi_d",
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
        MGXS(**kwargs)

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
        MGXS(**kwargs)

    assert expected_message in capsys.readouterr().out
