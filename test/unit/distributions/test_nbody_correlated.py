import numpy as np

import mcdc.transport.distribution as dist
from mcdc.object_.distribution import DistributionNBody


def test_nbody_sample_correlated(mock_rng_sequence, prepare_simulation):
    # MCNP Theory & User Manual §2.4.3.5.4.13 (Law 66: N-body Phase Space Distribution)
    distribution = DistributionNBody(
        values=[2.0, 4.0, 6.0],
        probabilities=[1.0, 1.0, 1.0],
    )
    simulation_container, data = prepare_simulation(objects=[distribution])
    simulation = simulation_container[0]
    distribution_base = simulation["distributions"][distribution.ID]

    # First value samples energy, second value samples isotropic cosine.
    xi1, xi2 = 0.2, 0.75
    mock_rng = mock_rng_sequence(xi1, xi2)

    sampled_E, sampled_mu = dist.sample_correlated_distribution(
        2.0,
        distribution_base,
        mock_rng,
        simulation,
        data,
    )

    # The current implementation samples energy from the tabulated distribution and
    # samples the cosine isotropically. This test is therefore checking the current
    # reduced implementation, not reconstructing the full Law 66 rejection sampler
    # from Eq. (2.103) through Eq. (2.106).
    # The constant PDF is normalized to 0.25 over [2, 6], so inverse-CDF sampling
    # in the first bin gives E_out = 2 + xi_1 / 0.25.
    # For the angular part, MCNP Eq. (2.107) gives mu = 2 * xi_10 - 1 for isotropic
    # center-of-mass sampling.
    expected_E = 2.0 + xi1 / 0.25
    expected_mu = 2.0 * xi2 - 1.0

    np.testing.assert_allclose(sampled_E, expected_E, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sampled_mu, expected_mu, rtol=0.0, atol=1e-12)
