import numpy as np
import mcdc as MCDC
from mcdc import type_
from mcdc.main import closeout
from mcdc.kernel import rng, AxV
import mcdc.global_ as mcdc_

input_deck = mcdc_.input_deck


def iqmc_dummy_mcdc_variable():
    """
    This function returns the global "mcdc" container. Inputs are
    taken from the kornreich eigenvalue problem.

    """
    # Set materials
    m1 = MCDC.material(
        capture=np.array([0.0]),
        scatter=np.array([[0.9]]),
        fission=np.array([0.1]),
        nu_p=np.array([6.0]),
    )
    m2 = MCDC.material(
        capture=np.array([0.68]),
        scatter=np.array([[0.2]]),
        fission=np.array([0.12]),
        nu_p=np.array([2.5]),
    )

    # Set surfaces
    s1 = MCDC.surface("plane-x", x=0.0, bc="vacuum")
    s2 = MCDC.surface("plane-x", x=1.5)
    s3 = MCDC.surface("plane-x", x=2.5, bc="vacuum")

    # Set cells
    MCDC.cell([+s1, -s2], m1)
    MCDC.cell([+s2, -s3], m2)

    # =============================================================================
    # iQMC Parameters
    # =============================================================================
    N = 100
    maxit = 10
    tol = 1e-3
    x = np.arange(0.0, 2.6, 0.1)
    Nx = len(x) - 1
    generator = "halton"
    solver = "power_iteration"
    fixed_source = np.zeros(Nx)
    phi0 = np.ones((Nx))

    # =============================================================================
    # Set tally, setting, and run mcdc
    # =============================================================================

    MCDC.iQMC(
        x=x,
        fixed_source=fixed_source,
        phi0=phi0,
        maxitt=maxit,
        tol=tol,
        generator=generator,
        eigenmode_solver=solver,
    )
    # Setting
    MCDC.setting(N_particle=N)
    MCDC.eigenmode()
    return MCDC.prepare()


def test_rn_basic():
    """
    Basic test routine for random number generator.
    Get the first 5 random numbers, skip a few, get 5 more and compare to
    reference data from [1]

    Seed numbers for index 1-5, 123456-123460

    [1] F. B. Brown, “Random number generation with arbitrary strides”,
        Trans. Am. Nucl. Soc, 71, 202 (1994)

    """
    MCDC.reset_cards()

    ref_data = np.array(
        (
            1,
            2806196910506780710,
            6924308458965941631,
            7093833571386932060,
            4133560638274335821,
        )
    )

    mcdc = iqmc_dummy_mcdc_variable()

    # run through the first five seeds (1-5)
    for i in range(5):
        assert mcdc["setting"]["rng_seed"] == ref_data[i]
        rng(mcdc["setting"])


def test_AxV_linearity():
    """
    AxV is the linear operator used for GMRES in iQMC.

    Linear operators must satisfy conditions of additivity and multiplicity
    defined as:
            - Additivity: f(x+y) = f(x) + f(y)
            - Multiplicity: f(cx) = cf(x)

    We can test both properties with:
            - f(a*x + b*y) = a*f(x) + b*f(y)
    """
    MCDC.reset_cards()

    mcdc = iqmc_dummy_mcdc_variable()

    size = mcdc["technique"]["iqmc"]["total_source"].size
    np.random.seed(123456)
    a = np.random.random()
    b = np.random.random()
    x = np.random.random((size,))
    y = np.random.random((size,))
    rhs = np.zeros((size,))

    F1 = AxV((a * x + b * y), rhs, mcdc)
    F2 = a * AxV(x, rhs, mcdc) + b * AxV(y, rhs, mcdc)
    assert np.allclose(F1, F2, rtol=1e-10)


# if __name__ == "__main__":
#     test_AxV_linearity()
