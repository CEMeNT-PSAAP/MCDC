import numpy as np

from mcdc.type_ import nuclide
from mcdc.input_ import nuclide


def generate_test_nuclear_data(n_neutron_groups, n_dnp_groups):
    # Set random seed
    np.random.seed(90053)

    # Generate nuclear data
    capture = np.random.rand(n_neutron_groups)
    scatter = np.random.rand(n_neutron_groups, n_neutron_groups)
    fission = np.random.rand(n_neutron_groups)
    nu_s = np.random.rand(n_neutron_groups)
    nu_p = np.random.rand(n_neutron_groups)
    nu_d = np.random.rand(n_dnp_groups, n_neutron_groups)
    chi_p = np.random.rand(n_neutron_groups, n_neutron_groups)
    chi_d = np.random.rand(n_neutron_groups, n_dnp_groups)
    speed = np.random.rand(n_neutron_groups)
    decay = np.random.rand(n_dnp_groups)

    return [capture, scatter, fission, nu_s, nu_p, nu_d, chi_p, chi_d, speed, decay]


def verify_card_quantities_with_input_quantities(
    mat, capture, scatter, fission, nu_s, nu_p, nu_d, chi_p, chi_d, speed, decay
):
    # Verify pass-through quantities are correct
    assert np.allclose(mat["capture"], capture)
    assert np.allclose(mat["fission"], fission)
    assert np.allclose(mat["nu_p"], nu_p)
    assert np.allclose(mat["nu_s"], nu_s)
    assert np.allclose(mat["speed"], speed)
    assert np.allclose(mat["decay"], decay)

    # Verify modified and/or calculated quantities are correct
    n_neutron_groups = len(capture)
    n_dnp_groups = len(decay)
    total_scatter = np.sum(scatter, 0)
    total = capture + total_scatter + fission
    nu_f = nu_p + np.sum(np.transpose(nu_d), 1)
    normalized_chi_s = scatter * np.divide(1.0, np.sum(scatter, 0))
    normalized_chi_p = chi_p * np.divide(1.0, np.sum(chi_p, 0))
    normalized_chi_d = chi_d * np.divide(1.0, np.sum(chi_d, 0))
    assert mat["G"] == n_neutron_groups
    assert mat["J"] == n_dnp_groups
    assert np.allclose(mat["scatter"], total_scatter)
    assert np.allclose(mat["total"], total)
    assert np.allclose(mat["nu_d"], np.transpose(nu_d))
    assert np.allclose(mat["nu_f"], nu_f)
    assert np.allclose(mat["chi_s"], np.transpose(normalized_chi_s))
    assert np.allclose(mat["chi_p"], np.transpose(normalized_chi_p))
    assert np.allclose(mat["chi_d"], np.transpose(normalized_chi_d))


def test_single_neutron_energy_and_zero_dnp_group_nuclide():
    n_neutron_groups = 1
    n_dnp_groups = 0

    # Generate data for one energy group and one DNP group
    [
        capture,
        scatter,
        fission,
        nu_s,
        nu_p,
        nu_d,
        chi_p,
        chi_d,
        speed,
        decay,
    ] = generate_test_nuclear_data(n_neutron_groups, n_dnp_groups)

    # Initialize nuclide with single energy and DNP group
    m = nuclide(
        capture=capture,
        scatter=scatter,
        fission=fission,
        nu_p=nu_p,
        nu_d=nu_d,
        chi_p=chi_p,
        chi_d=chi_d,
        nu_s=nu_s,
        speed=speed,
        decay=decay,
    )

    # Verify input processing
    verify_card_quantities_with_input_quantities(
        m, capture, scatter, fission, nu_s, nu_p, nu_d, chi_p, chi_d, speed, decay
    )


def test_single_neutron_energy_and_single_dnp_group_nuclide():
    n_neutron_groups = 1
    n_dnp_groups = 1

    # Generate data for one energy group and one DNP group
    [
        capture,
        scatter,
        fission,
        nu_s,
        nu_p,
        nu_d,
        chi_p,
        chi_d,
        speed,
        decay,
    ] = generate_test_nuclear_data(n_neutron_groups, n_dnp_groups)

    # Initialize nuclide with single energy and DNP group
    m = nuclide(
        capture=capture,
        scatter=scatter,
        fission=fission,
        nu_p=nu_p,
        nu_d=nu_d,
        chi_p=chi_p,
        chi_d=chi_d,
        nu_s=nu_s,
        speed=speed,
        decay=decay,
    )

    # Verify input processing
    verify_card_quantities_with_input_quantities(
        m, capture, scatter, fission, nu_s, nu_p, nu_d, chi_p, chi_d, speed, decay
    )


def test_multiple_neutron_energy_and_zero_dnp_group_nuclide():
    n_neutron_groups = 7
    n_dnp_groups = 0

    # Generate data for one energy group and one DNP group
    [
        capture,
        scatter,
        fission,
        nu_s,
        nu_p,
        nu_d,
        chi_p,
        chi_d,
        speed,
        decay,
    ] = generate_test_nuclear_data(n_neutron_groups, n_dnp_groups)

    # Initialize nuclide with single energy and DNP group
    m = nuclide(
        capture=capture,
        scatter=scatter,
        fission=fission,
        nu_p=nu_p,
        nu_d=nu_d,
        chi_p=chi_p,
        chi_d=chi_d,
        nu_s=nu_s,
        speed=speed,
        decay=decay,
    )

    # Verify input processing
    verify_card_quantities_with_input_quantities(
        m, capture, scatter, fission, nu_s, nu_p, nu_d, chi_p, chi_d, speed, decay
    )


def test_multiple_neutron_energy_and_single_dnp_group_nuclide():
    n_neutron_groups = 7
    n_dnp_groups = 1

    # Generate data for one energy group and one DNP group
    [
        capture,
        scatter,
        fission,
        nu_s,
        nu_p,
        nu_d,
        chi_p,
        chi_d,
        speed,
        decay,
    ] = generate_test_nuclear_data(n_neutron_groups, n_dnp_groups)

    # Initialize nuclide with single energy and DNP group
    m = nuclide(
        capture=capture,
        scatter=scatter,
        fission=fission,
        nu_p=nu_p,
        nu_d=nu_d,
        chi_p=chi_p,
        chi_d=chi_d,
        nu_s=nu_s,
        speed=speed,
        decay=decay,
    )

    # Verify input processing
    verify_card_quantities_with_input_quantities(
        m, capture, scatter, fission, nu_s, nu_p, nu_d, chi_p, chi_d, speed, decay
    )


def test_single_neutron_energy_and_multiple_dnp_group_nuclide():
    n_neutron_groups = 1
    n_dnp_groups = 6

    # Generate data for one energy group and one DNP group
    [
        capture,
        scatter,
        fission,
        nu_s,
        nu_p,
        nu_d,
        chi_p,
        chi_d,
        speed,
        decay,
    ] = generate_test_nuclear_data(n_neutron_groups, n_dnp_groups)

    # Initialize nuclide with single energy and DNP group
    m = nuclide(
        capture=capture,
        scatter=scatter,
        fission=fission,
        nu_p=nu_p,
        nu_d=nu_d,
        chi_p=chi_p,
        chi_d=chi_d,
        nu_s=nu_s,
        speed=speed,
        decay=decay,
    )

    # Verify input processing
    verify_card_quantities_with_input_quantities(
        m, capture, scatter, fission, nu_s, nu_p, nu_d, chi_p, chi_d, speed, decay
    )


def test_multiple_neutron_energy_and_multiple_dnp_group_nuclide():
    n_neutron_groups = 7
    n_dnp_groups = 6

    # Generate data for one energy group and one DNP group
    [
        capture,
        scatter,
        fission,
        nu_s,
        nu_p,
        nu_d,
        chi_p,
        chi_d,
        speed,
        decay,
    ] = generate_test_nuclear_data(n_neutron_groups, n_dnp_groups)

    # Initialize nuclide with single energy and DNP group
    m = nuclide(
        capture=capture,
        scatter=scatter,
        fission=fission,
        nu_p=nu_p,
        nu_d=nu_d,
        chi_p=chi_p,
        chi_d=chi_d,
        nu_s=nu_s,
        speed=speed,
        decay=decay,
    )

    # Verify input processing
    verify_card_quantities_with_input_quantities(
        m, capture, scatter, fission, nu_s, nu_p, nu_d, chi_p, chi_d, speed, decay
    )
